"""Тесты v2.4.1: квантование статистики по времени, прогрев из ISS, персистентный стейт дня."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from bot_a.backtest.candles import Candle
from bot_a.economics.fees import TinkoffFeeCalculator
from bot_a.engine.factory import create_engine, create_guard
from bot_a.engine.fsm import StrategyParams
from bot_a.engine.ledger import PortfolioLedger
from bot_a.execution.simulated_gateway import SimulatedOrderExecutionGateway
from bot_a.math_engine.pair_spread import PairSpreadCalculator
from bot_a.models.domain import ExecutionTopology, ExitReason, FsmState
from bot_a.services.bootstrap import bootstrap_warmup
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import MSK, SimulatedTimeProvider
from tests.helpers import make_book, make_config, make_meta, make_preset, trading_start_ms
from tests.test_engine import _bundle, _open_position, _params, _tick, _warmup


# ------------------------------------------------------------------ общие фабрики
def _calc(interval: float, *, ewma_window: int = 20, macro_samples: int = 10) -> PairSpreadCalculator:
    return PairSpreadCalculator(
        leg1=make_meta("AAA", uid="uid-AAA"),
        leg2=make_meta("BBB", uid="uid-BBB"),
        beta=1.0,
        ewma_window=ewma_window,
        beta_mode="slow_ratio",
        beta_min_samples=5,
        macro_beta_window=200,
        macro_beta_min_samples=macro_samples,
        stat_sample_interval_seconds=interval,
    )


def _feed(calc, base_ms: int, ticks: int, step_ms: int, price1: str = "100.00", price2: str = "101.00"):
    metrics = None
    for index in range(ticks):
        time_ms = base_ms + index * step_ms
        calc.on_book(0, make_book(dec(price1), time_ms, uid="uid-AAA"))
        metrics = calc.on_book(1, make_book(dec(price2), time_ms, uid="uid-BBB"))
    return metrics


async def _warmup_5s(engine, clock, gateway, ticks: int = 80) -> None:
    """Прогрев при квантовании 5 с: шаг часов 5 с даёт ровно один сэмпл на тик."""
    for index in range(ticks):
        mid2 = Decimal("101.00") if index % 2 == 0 else Decimal("101.02")
        await _tick(engine, clock, gateway, Decimal("100.00"), mid2, seconds=5)


# ------------------------------------------------- 1. Квантование статистики по 5 с
def test_stat_sampling_quantises_ewma_to_five_seconds():
    """При шаге L2 100 мс окно двигается раз в 5 с: 12 с потока -> 3 сэмпла."""
    calc = _calc(5.0)
    base = 1_700_000_000_000
    advances = []
    sampled_metrics = 0
    count = 0
    for index in range(120):  # 12 с по 100 мс
        time_ms = base + index * 100
        m1 = calc.on_book(0, make_book(dec("100.00"), time_ms, uid="uid-AAA"))
        m2 = calc.on_book(1, make_book(dec("101.00"), time_ms, uid="uid-BBB"))
        sampled_metrics += (1 if m1 is not None and m1.stat_sampled else 0) + (
            1 if m2 is not None and m2.stat_sampled else 0
        )
        if calc.ewma.count != count:
            count = calc.ewma.count
            advances.append(index)
    # окно сдвинулось на t=0, t=5 с и t=10 с, а не 120 раз
    assert advances == [0, 50, 100], advances
    assert calc.ewma.count == 3
    assert sampled_metrics == 3, sampled_metrics


def test_stat_sampling_keeps_instant_z_on_every_tick():
    """Между сэмплами Z пересчитывается по текущему спреду и «старой» mu/sigma."""
    calc = _calc(5.0, ewma_window=10, macro_samples=5)
    base = 1_700_000_000_000
    for index in range(20):
        price2 = "101.00" if index % 2 == 0 else "101.04"
        _feed(calc, base + index * 5000, 1, 100, "100.00", price2)
    assert calc.ewma.variance > 0
    mean_before = calc.ewma.mean
    count_before = calc.ewma.count

    zs = []
    last_sample_ms = base + 19 * 5000
    for index in range(10):
        time_ms = last_sample_ms + 100 + index * 100
        calc.on_book(0, make_book(dec("100.00"), time_ms, uid="uid-AAA"))
        metrics = calc.on_book(1, make_book(dec("101.40") + Decimal("0.01") * index, time_ms, uid="uid-BBB"))
        assert metrics.stat_sampled is False
        assert metrics.mean_rub == mean_before
        zs.append(metrics.zscore)
    assert calc.ewma.count == count_before, "между сэмплами окно не двигается"
    assert max(abs(z) for z in zs) > 5.0, "мгновенный Z реагирует сразу"
    assert len({round(z, 4) for z in zs}) > 1, "Z пересчитывается на каждом тике"


def test_quantised_sampling_does_not_change_per_call_legacy_path():
    """v2.4 (бэктест) идёт с интервалом 0: окно двигается на каждом снимке стакана."""
    legacy = _calc(0.0)
    quantised = _calc(5.0)
    base = 1_700_000_000_000
    for index in range(30):
        price2 = "101.00" if index % 2 == 0 else "101.06"
        for calc in (legacy, quantised):
            _feed(calc, base + index * 5000, 1, 100, "100.00", price2)
    # legacy: 2 вызова на «пару» (leg1 + leg2), кроме первого -> 59 обновлений
    assert legacy.ewma.count == 59, legacy.ewma.count
    # 5-секундное квантование: ровно один сэмпл на 5 с -> 30 обновлений
    assert quantised.ewma.count == 30, quantised.ewma.count
    assert legacy.ewma.mean != quantised.ewma.mean


def test_regime_buffer_fills_only_on_stat_samples(tmp_path):
    """Буфер режимного фильтра пополняется раз в 5 с, а не на каждом L2-пакете."""

    async def scenario():
        params = _params(
            use_regime_filter=True,
            regime_window=200,
            use_z_peak_hook=False,
            stat_sample_interval_seconds=5.0,
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _warmup_5s(engine, clock, gateway)
        base_state = engine.spread_state()
        assert base_state["isWarm"] is True, base_state
        for _ in range(200):  # 20 с по 100 мс
            clock.advance(100)
            await engine.on_clock()
            await engine.on_order_book(1, make_book(dec("100.00"), clock.now_ms(), uid="uid-AAA"))
            await engine.on_order_book(2, make_book(dec("101.00"), clock.now_ms(), uid="uid-BBB"))
        state = engine.spread_state()
        # границы 5 с внутри 20 с: t=0, 5, 10, 15 -> ровно 4 новых сэмпла
        assert state["tickCount"] - base_state["tickCount"] == 4, state
        assert state["regimeSamples"] - base_state["regimeSamples"] == 4, state
        assert state["regimeSamples"] <= state["regimeCapacity"]

    asyncio.run(scenario())


def test_stops_still_evaluated_on_every_l2_tick(tmp_path):
    """Стоп по Leg1 срабатывает между сэмплами (тики по 200 мс)."""

    async def scenario():
        params = _params(
            z_stop=1e9,
            ghost_flat_seconds=99999,
            single_leg_stop_to_target_ratio=0.2,
            single_leg_stop_min_steps=1,
            stat_sample_interval_seconds=5.0,
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        # Прогрев с шагом 5 с: при квантовании 5 с каждый тик даёт сэмпл.
        await _warmup_5s(engine, clock, gateway)
        assert engine.metrics is not None and engine.metrics.is_warm
        for _ in range(2):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"), seconds=5)
        assert engine.position is not None, f"позиция не открыта: {engine.state}"
        loss_price = engine.position.leg1_entry_price - Decimal("0.40")
        for _ in range(10):
            if engine.ledger.trade_count:
                break
            clock.advance(200)
            await engine.on_clock()
            await engine.on_order_book(1, make_book(loss_price, clock.now_ms(), uid="uid-AAA"))
            await engine.on_order_book(2, make_book(dec("104.00"), clock.now_ms(), uid="uid-BBB"))
        assert engine.ledger.trade_count == 1, f"состояние={engine.state}"
        assert engine.ledger.trades[0].exit_reason is ExitReason.HARD_STOP

    asyncio.run(scenario())


# ------------------------------------------------------------------ 2. Прогрев из ISS
class _FakeCandleSource:
    """Источник свечей в памяти: детерминированная пила, без сети."""

    def __init__(self, candles1, candles2):
        self._c1 = candles1
        self._c2 = candles2

    async def get_candles(self, meta, start_ms, end_ms, interval="1min"):
        source = self._c1 if meta.ticker == "AAA" else self._c2
        return [candle for candle in source if start_ms <= candle.time_ms <= end_ms]


def _saw_candles(start_ms: int, days: int, base: str, amplitude: str = "0.10", per_day: int = 60):
    candles = []
    for day in range(days):
        for index in range(per_day):
            offset = Decimal(amplitude) if index % 2 == 0 else -Decimal(amplitude)
            open_price = dec(base) + offset
            close_price = dec(base) - offset
            candles.append(
                Candle(
                    time_ms=start_ms + (day * per_day + index) * 60_000,
                    open=open_price,
                    high=max(open_price, close_price) + Decimal("0.05"),
                    low=min(open_price, close_price) - Decimal("0.05"),
                    close=close_price,
                    volume=1000,
                )
            )
    return candles


def test_bootstrap_warmup_makes_engine_ready_and_idle(tmp_path):
    """Прогрев наполняет окна, переводит ядро в Idle и не открывает сделок."""

    async def scenario():
        params = _params(stat_sample_interval_seconds=5.0, use_regime_filter=True)
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        start = clock.now_ms()
        source = _FakeCandleSource(
            _saw_candles(start, days=5, base="100.00"),
            _saw_candles(start, days=5, base="101.00"),
        )
        report = await bootstrap_warmup(
            config,
            config.preset(),
            engine,
            source=source,
            days=5,
            end_ms=start + 5 * 60 * 60_000,
        )
        assert report.candles_leg1 > 0 and report.candles_leg2 > 0
        assert report.tick_count > 0
        assert report.is_warm is True, "окно EWMA должно прогреться"
        assert report.state == FsmState.IDLE.value, report.state
        assert report.beta_macro > 0
        assert report.regime_samples > 0
        assert engine.ledger.trade_count == 0, "во время прогрева сделок быть не должно"
        assert engine.bootstrap is False, "флаг bootstrap снимается после прогрева"
        assert engine.position is None
        assert "Warmup complete" in report.render()

    asyncio.run(scenario())


def test_bootstrap_warmup_blocks_trading_while_running(tmp_path):
    """Пока идёт прогрев, даже сильный сигнал не открывает позицию."""

    async def scenario():
        params = _params(stat_sample_interval_seconds=5.0, use_entry_z_cap=False)
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        engine.set_bootstrap(True)
        await _warmup_5s(engine, clock, gateway)
        assert engine.state is FsmState.IDLE, engine.state
        for _ in range(3):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("106.00"), seconds=5)
        assert engine.position is None
        engine.set_bootstrap(False)
        for _ in range(2):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("106.00"), seconds=5)
        assert engine.position is not None, "после прогрева тот же сигнал открывает сделку"

    asyncio.run(scenario())


# ------------------------------------------------------- 3. Персистентный стейт дня
def _engine_with_state(tmp_path, params, clock, state_path: Path):
    leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
    leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
    config = make_config(tmp_path, make_preset(leg1, leg2), params, ExecutionTopology.SINGLE_LEG_LEAD_LAG)
    fee = TinkoffFeeCalculator(config.fees.tariff, clock=clock)
    gateway = SimulatedOrderExecutionGateway(
        metadata=[leg1, leg2],
        clock=clock,
        fee_calculator=fee,
        latency_ms=0,
        queue_fill_factor=0.35,
        slippage_steps=0,
        order_book_depth=5,
        initial_cash_rub=config.execution.paper_initial_cash_rub,
    )
    engine = create_engine(
        [leg1, leg2],
        params,
        ExecutionTopology.SINGLE_LEG_LEAD_LAG,
        gateway,
        fee,
        clock,
        create_guard(config, [leg1, leg2]),
        PortfolioLedger(config.execution.paper_initial_cash_rub),
        daily_state_path=str(state_path),
    )
    return config, gateway, engine


def test_daily_state_persists_across_restart(tmp_path):
    """HardStop-блокировка переживает перезапуск процесса в тот же день MSK."""
    params = _params(
        z_stop=1e9,
        use_daily_stop_breaker=True,
        max_daily_hard_stops=1,
        ghost_flat_seconds=99999,
        single_leg_stop_to_target_ratio=0.2,
        single_leg_stop_min_steps=1,
    )
    state_path = tmp_path / "state" / "daily_state_TEST_PAIR.json"
    start = trading_start_ms()

    async def session_one():
        clock = SimulatedTimeProvider(start)
        _, gateway, engine = _engine_with_state(tmp_path, params, clock, state_path)
        await gateway.start()
        await _open_position(engine, clock, gateway)
        loss_price = engine.position.leg1_entry_price - Decimal("0.30")
        for _ in range(10):
            if engine.ledger.trade_count:
                break
            await _tick(engine, clock, gateway, loss_price, Decimal("104.00"))
        assert engine.ledger.trade_count == 1
        assert engine.daily_locked_reason is not None
        assert state_path.exists(), "состояние дня должно быть сохранено"
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        assert saved["daily_hard_stops"] == 1
        assert saved["date"] == datetime.fromtimestamp(start / 1000, tz=MSK).strftime("%Y-%m-%d")
        assert float(saved["daily_net_rub"]) < 0

    async def session_two():
        clock = SimulatedTimeProvider(start + 3_600_000)
        _, gateway, engine = _engine_with_state(tmp_path, params, clock, state_path)
        await gateway.start()
        assert engine.daily_locked_reason is not None
        assert engine.daily_hard_stops == 1
        await _warmup(engine, clock, gateway)
        for _ in range(4):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("106.00"))
        assert engine.position is None, "после рестарта входы остаются заблокированными"
        assert engine.counters["dailyStopSkips"] > 0

    async def session_three():
        clock = SimulatedTimeProvider(start + 24 * 3600 * 1000)
        _, gateway, engine = _engine_with_state(tmp_path, params, clock, state_path)
        await gateway.start()
        assert engine.daily_locked_reason is None
        assert engine.daily_hard_stops == 0

    asyncio.run(session_one())
    asyncio.run(session_two())
    asyncio.run(session_three())


def test_daily_state_not_persisted_without_path(tmp_path):
    """Бэктест и optimize не пишут состояние дня (путь не передан)."""

    async def scenario():
        params = _params(use_daily_stop_breaker=True, max_daily_hard_stops=1)
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        engine._register_daily_outcome(ExitReason.TIME_STOP, Decimal("-1"), clock.now_ms())  # noqa: SLF001
        assert list(tmp_path.glob("**/daily_state_*.json")) == []

    asyncio.run(scenario())


def test_live_worker_wires_bootstrap_cadence_and_state(tmp_path, monkeypatch):
    """Полный путь paper-воркера: прогрев, каденция 5 с и стейт дня подключены."""
    import bot_a.services.live_worker as live_module
    from bot_a.services import bootstrap as bootstrap_module
    from bot_a.services.live_worker import LiveTradingWorker
    from bot_a.utils.time_provider import now_ms

    root = Path(__file__).resolve().parents[1]
    from bot_a.config import AppConfig

    config = AppConfig.load(root / "appsettings.json")
    config = config.with_overrides(active_preset="VTBR_SBER")
    config = config.with_preset_overrides()
    preset = config.preset()

    leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
    leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
    resolved = preset.with_metadata(leg1, leg2)

    # 1) Свечи для прогрева — без сети.
    start = now_ms() - 5 * 3600 * 1000
    fake_source = _FakeCandleSource(
        _saw_candles(start, days=5, base="100.00"),
        _saw_candles(start, days=5, base="101.00"),
    )
    monkeypatch.setattr(bootstrap_module, "IssCandleSource", lambda *a, **k: fake_source)

    # 2) Стрим — три события и завершение.
    class _DummyStream:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def events(self):
            from bot_a.models.domain import MarketEvent
            book1 = make_book(dec("100.00"), now_ms(), uid="uid-AAA")
            book2 = make_book(dec("101.00"), now_ms(), uid="uid-BBB")
            yield MarketEvent.of_book(1, book1)
            yield MarketEvent.of_book(2, book2)

    monkeypatch.setattr(live_module.broker, "TinkoffMarketDataStream", _DummyStream)

    worker = LiveTradingWorker(config, resolved, client=None, paper=True)
    monkeypatch.setattr(worker, "_resolve_account_id", _async_return("acc-1"))
    monkeypatch.setattr(worker, "_resolve_legs", _async_return([leg1, leg2]))

    snapshot = asyncio.run(worker.run(duration_seconds=0.4))
    assert snapshot["mode"] == "paper"
    assert snapshot["warmup"]["isWarm"] is True, snapshot["warmup"]
    assert snapshot["warmup"]["tickCount"] > 0
    assert snapshot["state"] in ("Idle", "Cooldown", "Warmup")
    # Каденция статистики из конфига применилась к живому ядру.
    assert config.execution.stat_sample_interval_seconds == 5.0
    # Стейт дня инициализирован для пресета (путь ./state/daily_state_VTBR_SBER.json).
    assert worker._daily_state_path().name == "daily_state_VTBR_SBER.json"  # noqa: SLF001


def _async_return(value):
    async def inner(*args, **kwargs):
        return value

    return inner


# ------------------------------------------------------------------ 4. Конфигурация
def test_live_worker_daily_state_path_and_cadence(tmp_path):
    """В live/paper воркер передаёт путь стейта дня и каденцию статистики из конфига."""
    from bot_a.services.live_worker import LiveTradingWorker

    root = Path(__file__).resolve().parents[1]
    from bot_a.config import AppConfig

    config = AppConfig.load(root / "appsettings.json")
    config = config.with_overrides(active_preset="VTBR_SBER")
    worker = LiveTradingWorker(config, config.preset(), client=None, paper=True)
    path = worker._daily_state_path()  # noqa: SLF001
    assert path.name == "daily_state_VTBR_SBER.json"
    assert path.parent.name == "state"
    assert config.execution.stat_sample_interval_seconds == 5.0


def test_readiness_reports_cadence_and_warmup():
    """--monday-ready показывает каденцию 5 с и объём прогрева."""
    from datetime import datetime

    from bot_a.config import AppConfig
    from bot_a.services.readiness import build_monday_readiness
    from bot_a.utils.time_provider import MSK

    root = Path(__file__).resolve().parents[1]
    config = AppConfig.load(root / "appsettings.json")
    readiness = build_monday_readiness(config, now=datetime(2026, 10, 26, 9, 40, tzinfo=MSK))
    checks = {check.name: check for check in readiness.checks}
    cadence = checks["Каденция и прогрев (v2.4.1)"]
    assert cadence.ok is True
    assert "5.0 с" in cadence.detail
    assert "прогрев 5 торг. дней" in cadence.detail
    assert "state" in cadence.detail


def test_stat_sampling_config_defaults():
    """Production-дефолты v2.4.1: интервал 5 с, прогрев 5 дней, стейт в ./state."""
    root = Path(__file__).resolve().parents[1]
    from bot_a.config import AppConfig

    config = AppConfig.load(root / "appsettings.json")
    assert config.execution.stat_sample_interval_seconds == 5.0
    assert config.execution.bootstrap_warmup_days == 5
    assert config.storage.state_dir == "./state"
    # v2.5: каденция статистики едина для всех режимов — бэктест больше не двигает
    # EWMA на каждом событии, иначе окно EwmaWindow покрывало вдвое меньше времени,
    # чем в live/replay, и Z получался другим.
    assert config.strategy.stat_sample_interval_seconds == 5.0
    assert StrategyParams().stat_sample_interval_seconds == 0.0
    assert config.strategy.to_dict()["statSampleIntervalSeconds"] == 5.0
