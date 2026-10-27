"""Тесты v2.4: дневной предохранитель, макро-дрейф беты и фильтр σ/Fee."""

from __future__ import annotations

import asyncio
from decimal import Decimal

from bot_a.math_engine.pair_spread import PairSpreadCalculator
from bot_a.models.domain import ExecutionTopology, ExitReason, FsmState
from tests.helpers import make_book, make_meta
from tests.test_engine import _bundle, _open_position, _params, _tick, _warmup


# ---------------------------------------------------------------- F1: дневной стоп
def test_daily_stop_breaker_blocks_reentry_after_hard_stop(tmp_path):
    """После HardStop пара не открывает сделки до следующего торгового дня."""

    async def scenario():
        params = _params(
            z_stop=1e9,
            use_daily_stop_breaker=True,
            max_daily_hard_stops=1,
            ghost_flat_seconds=99999,
            single_leg_stop_to_target_ratio=0.2,
            single_leg_stop_min_steps=1,
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        entry = position.leg1_entry_price
        loss_price = entry - Decimal("0.30")
        for _ in range(6):
            if engine.ledger.trade_count:
                break
            await _tick(engine, clock, gateway, loss_price, Decimal("104.00"))
        assert engine.ledger.trade_count == 1, f"состояние={engine.state}"
        assert engine.ledger.trades[0].exit_reason is ExitReason.HARD_STOP
        assert engine.daily_locked_reason is not None
        assert engine.counters["dailyStopSkips"] == 0

        # Новый сигнал в тот же день: вход должен быть заблокирован.
        for _ in range(4):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("106.00"))
        assert engine.position is None, "вход в день HardStop запрещён"
        assert engine.counters["dailyStopSkips"] > 0
        assert engine.daily_hard_stops == 1

        # Следующий торговый день: блокировка снимается.
        clock.set(clock.now_ms() + 24 * 3600 * 1000)
        await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("101.00"))
        assert engine.daily_locked_reason is None
        assert engine.daily_hard_stops == 0

    asyncio.run(scenario())


def test_daily_loss_limit_in_fee_multiples_blocks_day(tmp_path):
    """Дневной убыток ниже −3.5 × комиссия круга закрывает день без HardStop."""

    async def scenario():
        params = _params(
            z_stop=1e9,
            use_daily_stop_breaker=True,
            max_daily_hard_stops=99,
            max_daily_loss_fee_multiple=Decimal("2.5"),
            ghost_flat_seconds=99999,
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        fee_rt = engine._current_round_trip_fee(1)  # noqa: SLF001
        assert fee_rt > 0
        # Каждый убыток = −1.5 комиссии; лимит −2.5 комиссии => блокировка на втором.
        engine._register_daily_outcome(  # noqa: SLF001
            ExitReason.TIME_STOP, -fee_rt * Decimal("1.5"), clock.now_ms()
        )
        assert engine.daily_locked_reason is None, "убыток ещё в пределах лимита"
        engine._register_daily_outcome(  # noqa: SLF001
            ExitReason.TIME_STOP, -fee_rt * Decimal("1.5"), clock.now_ms()
        )
        assert engine.daily_locked_reason is not None
        assert "дневной убыток" in engine.daily_locked_reason

    asyncio.run(scenario())


def test_daily_stop_breaker_disabled_allows_same_day_reentry(tmp_path):
    """При useDailyStopBreaker=False блокировки нет (A/B-совместимость)."""

    async def scenario():
        params = _params(
            z_stop=1e9,
            use_daily_stop_breaker=False,
            ghost_flat_seconds=99999,
            single_leg_stop_to_target_ratio=0.2,
            single_leg_stop_min_steps=1,
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        loss_price = engine.position.leg1_entry_price - Decimal("0.30")
        for _ in range(6):
            if engine.ledger.trade_count:
                break
            await _tick(engine, clock, gateway, loss_price, Decimal("104.00"))
        assert engine.ledger.trade_count == 1
        assert engine.daily_locked_reason is None

    asyncio.run(scenario())


# ---------------------------------------------------------------- F2: макро-дрейф беты
def test_macro_beta_drift_computed_and_ready():
    """β_macro считается всегда, дрейф появляется только после прогрева базы."""
    leg1 = make_meta("AAA", uid="uid-AAA")
    leg2 = make_meta("BBB", uid="uid-BBB")
    calc = PairSpreadCalculator(
        leg1=leg1,
        leg2=leg2,
        beta=1.0,
        ewma_window=20,
        beta_mode="slow_ratio",
        beta_min_samples=5,
        macro_beta_window=200,
        macro_beta_min_samples=100,
    )
    metrics = None
    for index in range(60):
        price2 = Decimal("101.00") + Decimal("0.01") * index
        metrics = calc.on_book(0, make_book(Decimal("100.00"), 1000 * index, uid="uid-AAA"))
        metrics = calc.on_book(1, make_book(price2, 1000 * index, uid="uid-BBB"))
    assert metrics is not None
    assert calc.beta_macro > 0
    assert metrics.beta_drift is None, "база ещё не прогрета до 100 наблюдений"
    for index in range(60, 160):
        price2 = Decimal("101.00") + Decimal("0.05") * index  # резкий рост Leg2
        metrics = calc.on_book(0, make_book(Decimal("100.00"), 1000 * index, uid="uid-AAA"))
        metrics = calc.on_book(1, make_book(price2, 1000 * index, uid="uid-BBB"))
    assert metrics is not None and metrics.beta_drift is not None
    assert metrics.beta_drift > 0.008, "Leg2 растёт быстрее -> дрейф положительный"
    assert metrics.beta_macro == calc.beta_macro


def test_macro_beta_drift_blocks_long_and_short(tmp_path):
    """Дрейф > +0.8 % блокирует Long Leg1, < −0.8 % — Short Leg1."""

    async def scenario():
        params = _params(
            use_macro_beta_drift_guard=True,
            max_beta_drift=0.008,
            ghost_flat_seconds=99999,
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _warmup(engine, clock, gateway)

        class _Stub:
            pass

        metrics = engine.metrics
        base = dict(
            time_ms=clock.now_ms(),
            spread_rub=metrics.spread_rub,
            mean_rub=metrics.mean_rub,
            variance=metrics.variance,
            zscore=metrics.zscore,
            obi_leg1=0.0,
            obi_leg2=0.0,
            micro_leg1=metrics.micro_leg1,
            micro_leg2=metrics.micro_leg2,
            best_bid_leg1=metrics.best_bid_leg1,
            best_ask_leg1=metrics.best_ask_leg1,
            best_bid_leg2=metrics.best_bid_leg2,
            best_ask_leg2=metrics.best_ask_leg2,
            is_warm=True,
            tick_count=metrics.tick_count,
            beta=metrics.beta,
        )
        from dataclasses import replace

        long_with_drift = replace(metrics, **{**base, "zscore": 3.0, "beta_drift": 0.02})
        await engine._maybe_enter(long_with_drift)  # noqa: SLF001
        assert engine.position is None
        assert engine.counters["betaDriftSkips"] == 1

        short_with_drift = replace(metrics, **{**base, "zscore": -3.0, "beta_drift": -0.02})
        await engine._maybe_enter(short_with_drift)  # noqa: SLF001
        assert engine.position is None
        assert engine.counters["betaDriftSkips"] == 2

        # Тот же сигнал, но дрейф в допустимой зоне — вход не блокируется дрейфом.
        calm_long = replace(metrics, **{**base, "zscore": 3.0, "beta_drift": 0.001})
        await engine._maybe_enter(calm_long)  # noqa: SLF001
        assert engine.counters["betaDriftSkips"] == 2
        assert engine.state in (FsmState.QUOTING_ENTRY_LEG1, FsmState.IDLE, FsmState.COOLDOWN)

    asyncio.run(scenario())


def test_macro_beta_drift_inert_for_same_underlying(tmp_path):
    """Для same_underlying бета постоянна, поэтому дрейф не мешает входам."""
    leg1 = make_meta("AAA", uid="uid-AAA")
    leg2 = make_meta("BBB", uid="uid-BBB")
    calc = PairSpreadCalculator(
        leg1=leg1,
        leg2=leg2,
        beta=1.0,
        ewma_window=20,
        beta_mode="same_underlying",
        macro_beta_window=40,
        macro_beta_min_samples=5,
    )
    for index in range(30):
        calc.on_book(0, make_book(Decimal("100.00"), 1000 * index, uid="uid-AAA"))
        metrics = calc.on_book(1, make_book(Decimal("100.00"), 1000 * index, uid="uid-BBB"))
    assert metrics.beta_drift is not None
    assert abs(metrics.beta_drift) < 1e-9


# ---------------------------------------------------------------- F3: σ/Fee
def test_vol_to_fee_filter_blocks_tight_volatility(tmp_path):
    """При σ_leg1 / Fee ниже порога вход отменяется, при высокой σ — проходит."""

    async def scenario():
        params = _params(
            use_vol_to_fee_filter=True,
            min_sigma_to_fee_ratio=1.30,
            ghost_flat_seconds=99999,
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _warmup(engine, clock, gateway)

        from dataclasses import replace

        metrics = engine.metrics
        baseline = engine.counters["volFilterSkips"]
        # σ искусственно сжимаем до нуля: фильтр должен сработать.
        tight = replace(metrics, variance=1e-12, zscore=3.0)
        await engine._maybe_enter(tight)  # noqa: SLF001
        assert engine.position is None
        assert engine.counters["volFilterSkips"] == baseline + 1

        # σ в 10 раз выше комиссии — вход не блокируется по этому фильтру.
        wide = replace(metrics, variance=400.0, zscore=3.0)
        await engine._maybe_enter(wide)  # noqa: SLF001
        assert engine.counters["volFilterSkips"] == baseline + 1

    asyncio.run(scenario())


def test_vol_to_fee_filter_disabled_by_default_in_engine_tests(tmp_path):
    """Фильтры v2.4 можно отключать по отдельности (A/B-совместимость)."""
    params = _params(use_vol_to_fee_filter=False, use_macro_beta_drift_guard=False)
    assert params.use_vol_to_fee_filter is False
    assert params.use_macro_beta_drift_guard is False
    assert params.to_dict()["minSigmaToFeeRatio"] == 1.15
    assert params.to_dict()["maxDailyHardStops"] == 1
