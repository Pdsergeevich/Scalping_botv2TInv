"""Тесты v2.5: защита от фантомных выбросов Z и подтверждение сигнала двумя замерами.

Сценарий из живого дня 26.10.2026: в 18:55:34 REST-цикл отдал пару, в которой одна нога
обновилась, а вторая была устаревшей (таймаут 20 с), и `Z` скакнул в `−2.298` при спокойном
рынке. Оба механизма ниже должны такие тики отбрасывать.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from bot_a.models.domain import ExecutionTopology
from bot_a.services.l2_sync import PairBookSyncer
from tests.helpers import make_book, make_config
from tests.test_engine import _bundle, _params, _tick, _warmup
from tests.test_v241 import _warmup_5s


def _book(price: str, time_ms: int, uid: str):
    return make_book(Decimal(price), time_ms, uid=uid)


# --------------------------------------------------- 1. Синхронизатор стаканов
def test_syncer_accepts_synchronised_pair():
    syncer = PairBookSyncer(interval_ms=300)
    fresh = {1: _book("100.00", 1_000, "uid-AAA"), 2: _book("101.00", 1_010, "uid-BBB")}
    decision = syncer.check(fresh, {1: 5_000.0, 2: 5_120.0})
    assert decision.ok is True, decision.reason
    assert syncer.skew_skips == 0


def test_syncer_rejects_when_one_leg_failed():
    """Таймаут одной ноги — тик нельзя пускать в расчёт Z (случай 18:55:34)."""
    syncer = PairBookSyncer(interval_ms=300)
    fresh = {1: _book("100.00", 1_000, "uid-AAA")}
    decision = syncer.check(fresh, {1: 5_000.0}, failed=[2])
    assert decision.ok is False
    assert "не ответила" in decision.reason
    assert syncer.skew_skips == 1


def test_syncer_rejects_big_exchange_timestamp_skew():
    """Устаревшая книга от биржи: метки ног разошлись сильнее 1.5 интервала опроса."""
    syncer = PairBookSyncer(interval_ms=300)
    fresh = {1: _book("100.00", 1_000, "uid-AAA"), 2: _book("101.00", 21_000, "uid-BBB")}
    decision = syncer.check(fresh, {1: 5_000.0, 2: 5_050.0})
    assert decision.ok is False
    assert "биржевые метки" in decision.reason
    assert decision.skew_ms >= 20_000
    assert syncer.last_skew_ms >= 20_000


def test_syncer_rejects_big_request_skew():
    """Ноги опрошены с разрывом больше 1.5 интервала — пара не синхронна по времени."""
    syncer = PairBookSyncer(interval_ms=300)
    fresh = {1: _book("100.00", 1_000, "uid-AAA"), 2: _book("101.00", 1_000, "uid-BBB")}
    decision = syncer.check(fresh, {1: 0.0, 2: 900.0})
    assert decision.ok is False
    assert "запросы разошлись" in decision.reason


def test_syncer_rejects_empty_book():
    """После закрытия сессии биржа отдаёт пустой стакан — такой тик считать нельзя."""
    from bot_a.models.domain import OrderBookSnapshot

    syncer = PairBookSyncer(interval_ms=300)
    empty = OrderBookSnapshot(instrument_uid="uid-BBB", time_ms=1_000, bids=(), asks=(), last_price=None)
    fresh = {1: _book("100.00", 1_000, "uid-AAA"), 2: empty}
    decision = syncer.check(fresh, {1: 0.0, 2: 0.0})
    assert decision.ok is False
    assert "пустой стакан" in decision.reason
    assert syncer.skew_skips == 1


def test_syncer_rejects_frozen_book_while_other_moves():
    """Биржа отдаёт кэш: книга одной ноги не меняется, пока вторая двигается."""
    syncer = PairBookSyncer(interval_ms=300, stale_cycles=3)
    ok_count = 0
    for step in range(6):
        fresh = {
            1: _book("100.00", 1_000, "uid-AAA"),
            2: _book(f"101.{step:02d}", 1_000 + step * 100, "uid-BBB"),
        }
        if syncer.check(fresh, {1: 0.0, 2: 0.0}).ok:
            ok_count += 1
    assert ok_count <= 3, "замороженная нога должна перестать пропускаться"
    assert syncer.skew_skips >= 3


# ------------------------------------- 2. Фантомный выброс не доходит до ядра
async def _feed_pair_cycle(engine, clock, mid1: Decimal, mid2: Decimal, time_ms: int) -> None:
    clock.set(time_ms)
    await engine.on_clock()
    await engine.on_order_book(1, make_book(mid1, time_ms, uid="uid-AAA"))
    await engine.on_order_book(2, make_book(mid2, time_ms, uid="uid-BBB"))


def test_phantom_spike_filtered_by_dispatcher(tmp_path):
    """Скачок спреда при рассинхроне ног не доходит до ядра (случай 18:55:34).

    Сравниваем два сценария на одинаковом потоке: «наивный» диспетчер кормит ядро
    каждой ногой по отдельности (так было до v2.5), защищённый — только парами,
    прошедшими ``PairBookSyncer``. Фантомный тик обязан исчезнуть.
    """
    from bot_a.utils.time_provider import MSK, msk_to_ms
    from datetime import datetime

    base = msk_to_ms(datetime(2026, 10, 26, 15, 0, tzinfo=MSK))

    async def naive_max_z() -> float:
        params = _params()
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        for step in range(80):
            mid2 = Decimal("101.00") if step % 2 == 0 else Decimal("101.02")
            await _feed_pair_cycle(engine, clock, Decimal("100.00"), mid2, base + step * 300)
        # нога 1 уходит на 30 копеек, нога 2 «зависла» на старой цене
        await engine.on_order_book(1, make_book(Decimal("100.30"), base + 80 * 300, uid="uid-AAA"))
        return abs(float(engine.metrics.zscore))

    async def guarded_max_z() -> float:
        params = _params()
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        syncer = PairBookSyncer(interval_ms=300)
        for step in range(80):
            mid2 = Decimal("101.00") if step % 2 == 0 else Decimal("101.02")
            await _feed_pair_cycle(engine, clock, Decimal("100.00"), mid2, base + step * 300)
        now = base + 80 * 300
        fresh = {
            1: make_book(Decimal("100.30"), now, uid="uid-AAA"),
            2: make_book(Decimal("101.00"), now - 20_000, uid="uid-BBB"),  # книга 20 с назад
        }
        decision = syncer.check(fresh, {1: now, 2: now + 40})
        assert decision.ok is False and "биржевые метки" in decision.reason
        assert syncer.skew_skips == 1
        return abs(float(engine.metrics.zscore))

    naive = asyncio.run(naive_max_z())
    guarded = asyncio.run(guarded_max_z())
    assert naive > 3.0, f"наивное кормление обязано дать фантомный выброс, получено {naive}"
    assert guarded < 1.0, f"защищённое кормление не должно давать выброс, получено {guarded}"
    assert guarded < naive / 2


# ------------------------------------- 3. Подтверждение сигнала двумя замерами
def test_single_sample_signal_is_not_confirmed(tmp_path):
    """Один статистический замер |Z| >= Z_entry не армирует пик и не даёт сигнала."""

    async def scenario():
        params = _params(
            use_entry_persistence=True, entry_confirm_samples=2, stat_sample_interval_seconds=5.0
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await _warmup_5s(engine, clock, gateway)
        before = dict(engine.counters)
        engine._z_peak = 0.0  # noqa: SLF001 — чистый старт проверки пика
        await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"), seconds=5)
        assert engine.counters["persistenceSkips"] > before["persistenceSkips"], engine.counters
        assert engine.counters["signals"] == before["signals"], "сигнала быть не должно"
        assert engine.position is None
        assert engine._z_peak == 0.0, "одиночный выброс не должен армировать пик"  # noqa: SLF001

    asyncio.run(scenario())


def test_two_consecutive_samples_are_confirmed(tmp_path):
    """Два подряд статистических замера |Z| >= Z_entry проходят фильтр подтверждения."""

    async def scenario():
        params = _params(
            use_entry_persistence=True, entry_confirm_samples=2, stat_sample_interval_seconds=5.0
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await _warmup_5s(engine, clock, gateway)
        before = dict(engine.counters)
        engine._z_peak = 0.0  # noqa: SLF001 — чистый старт проверки пика
        for _ in range(3):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"), seconds=5)
        assert engine.counters["persistenceSkips"] > before["persistenceSkips"], engine.counters
        assert engine.counters["signals"] > before["signals"], "подтверждённый сигнал обязан дойти"
        assert engine._z_peak >= params.z_entry  # noqa: SLF001

    asyncio.run(scenario())


def test_persistence_disabled_by_default(tmp_path):
    """По умолчанию (бэктест) поведение прежнее: сигнал с первого замера."""

    async def scenario():
        params = _params()
        assert params.use_entry_persistence is False
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await _warmup(engine, clock, gateway)
        await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"))
        assert engine.position is not None, "без подтверждения вход идёт с первого тика"
        assert engine.counters["persistenceSkips"] == 0

    asyncio.run(scenario())


# ------------------------------------- 4. Мульти-парный раннер (`--preset ALL_ACTIVE`)
class _MultiFakeClient:
    """REST-клиент для двух пар: два инструмента с небольшим дрейфом цен."""

    def __init__(self) -> None:
        self.index = 0
        self.base = 1_700_000_000_000

    async def resolve_instrument(self, ticker: str, class_code: str = "TQBR", market: Any = None):
        from tests.helpers import make_meta

        return make_meta(ticker, price_step="0.01", step_amount="0.01", uid=f"uid-{ticker}")

    async def get_order_book(self, instrument_uid: str, depth: int = 20):
        prices = {
            "uid-AAA": Decimal("100.00"),
            "uid-BBB": Decimal("200.00"),
            "uid-CCC": Decimal("50.00"),
            "uid-DDD": Decimal("75.00"),
        }
        # Обе ноги одного цикла опроса получают одинаковую метку времени —
        # как в реальном диспетчере, который кормит ядро только синхронными парами.
        if instrument_uid in ("uid-AAA", "uid-CCC"):
            self.index += 1
        base = prices.get(instrument_uid, Decimal("100.00"))
        shift = Decimal("0.01") * Decimal(self.index % 30)
        return make_book(base + shift, self.base + self.index * 300, uid=instrument_uid)

    async def get_last_trades(self, instrument_uid: str, since=None, date_to=None):
        return []


def test_multi_preset_runner_tracks_every_pair(tmp_path):
    """ALL_ACTIVE ведёт обе пары независимо и публикует сводку по каждой."""
    from dataclasses import replace

    from bot_a.config import PairPreset
    from bot_a.services.multi_preset import MultiPresetRunner
    from tests.helpers import make_meta, make_preset
    from tests.test_watch import _no_warmup
    from tests.test_engine import _params

    async def scenario():
        leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
        leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
        leg3 = make_meta("CCC", price_step="0.01", step_amount="0.01", uid="uid-CCC")
        leg4 = make_meta("DDD", price_step="0.01", step_amount="0.01", uid="uid-DDD")
        first = make_preset(leg1, leg2)
        second = PairPreset(name="CCC_DDD", description="вторая пара", leg1=leg3, leg2=leg4)
        config = _no_warmup(make_config(tmp_path, first, _params()))
        config = replace(
            config,
            presets={"AAA_BBB": replace(first, name="AAA_BBB"), "CCC_DDD": second},
        )
        runner = MultiPresetRunner(
            config,
            [replace(first, name="AAA_BBB"), second],
            _MultiFakeClient(),
            interval_ms=100,
            depth=5,
            serve_dashboard=False,
        )
        stop = asyncio.Event()

        async def stopper():
            await asyncio.sleep(0.8)
            stop.set()

        asyncio.create_task(stopper())
        result = await runner.run(duration_seconds=None, stop_event=stop)
        assert len(result["pairs"]) == 2, result
        names = {pair["name"] for pair in result["pairs"]}
        assert names == {"AAA_BBB", "CCC_DDD"}
        for pair in result["pairs"]:
            assert pair["snapshots"] > 0, pair
            assert pair["z"] is not None
            assert pair["sigmaToFee"] is not None
            assert "day" in pair and "maxAbs" in pair["day"]
            assert pair["restErrors"] == 0

    asyncio.run(scenario())
