"""Проверки FSM v2: profit-lock, frozen mu, ghost-reversion, v1-совместимость."""

import asyncio
from decimal import Decimal

from bot_a.economics.fees import TinkoffFeeCalculator
from bot_a.engine.factory import create_engine, create_guard
from bot_a.engine.fsm import ROLE_EXIT_LEG1, ROLE_TP_LEG1, StrategyParams
from bot_a.engine.ledger import PortfolioLedger
from bot_a.execution.simulated_gateway import SimulatedOrderExecutionGateway
from bot_a.models.domain import ExecutionTopology, ExitReason, FsmState
from bot_a.utils.time_provider import SimulatedTimeProvider
from tests.helpers import make_book, make_config, make_meta, make_preset, make_trade, trading_start_ms


def _bundle(tmp_path, topology: ExecutionTopology, params: StrategyParams):
    leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
    leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
    preset = make_preset(leg1, leg2)
    config = make_config(tmp_path, preset, params, topology)
    clock = SimulatedTimeProvider(trading_start_ms())
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
    guard = create_guard(config, [leg1, leg2])
    ledger = PortfolioLedger(config.execution.paper_initial_cash_rub)
    engine = create_engine([leg1, leg2], params, topology, gateway, fee, clock, guard, ledger)
    return config, clock, fee, gateway, engine


async def _tick(engine, clock, gateway, mid1: Decimal, mid2: Decimal, seconds: int = 1) -> None:
    clock.advance(seconds * 1000)
    await engine.on_clock()
    book1 = make_book(mid1, clock.now_ms(), uid="uid-AAA")
    book2 = make_book(mid2, clock.now_ms(), uid="uid-BBB")
    await engine.on_order_book(1, book1)
    await engine.on_order_book(2, book2)
    await engine.on_trade(1, make_trade(book1.bids[0].price, clock.now_ms(), quantity=50, uid="uid-AAA"))
    await engine.on_trade(2, make_trade(book2.bids[0].price, clock.now_ms(), quantity=50, uid="uid-BBB"))


async def _warmup(engine, clock, gateway, ticks: int = 80) -> None:
    """Тёплый старт с ненулевой дисперсией спреда (иначе sigma = 0)."""
    for index in range(ticks):
        mid2 = Decimal("101.00") if index % 2 == 0 else Decimal("101.02")
        await _tick(engine, clock, gateway, Decimal("100.00"), mid2)


async def _open_position(engine, clock, gateway, shock: str = "104") -> None:
    await _warmup(engine, clock, gateway)
    for _ in range(2):
        await _tick(engine, clock, gateway, Decimal("100.00"), Decimal(shock))
    assert engine.position is not None, f"позиция не открыта: {engine.state}"
    assert engine.position.leg1_filled_qty == 1


def _params(**overrides) -> StrategyParams:
    base = StrategyParams(
        z_entry=1.0,
        z_exit=0.2,
        z_cancel=0.1,
        z_stop=100.0,
        ewma_window=40,
        min_edge_multiplier=Decimal("1.0"),
        slippage_buffer_steps=0,
        requote_cooldown_ms=0,
        max_hold_seconds=600,
        max_exit_quote_seconds=5,
        cooldown_seconds=1,
        quantity_lots=1,
        ghost_flat_seconds=45,
        scratch_seconds=60,
        # Механические тесты v2/v2.1 проверяют конкретные ветки ФСМ: фильтры v2.2
        # выключены, для них есть отдельные тесты.
        use_z_peak_hook=False,
        use_regime_filter=False,
        use_tp_decay=False,
        use_breakeven_ratchet=False,
        # v2.4: режимные предохранители тестируются отдельно в test_v24.py.
        use_daily_stop_breaker=False,
        use_macro_beta_drift_guard=False,
        use_vol_to_fee_filter=False,
    )
    return base.with_overrides(**overrides) if overrides else base


def test_resting_profit_locked_tp_closes_with_gain(tmp_path):
    """Правило 2: пассивный тейк на прибыльном уровне закрывает сделку в плюс."""

    async def scenario():
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, _params()
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        tp_order_id = engine._roles.get(ROLE_TP_LEG1)  # noqa: SLF001
        assert tp_order_id is not None, "прибыльный тейк должен стоять в стакане"
        tp_price = gateway.order(tp_order_id).price
        assert tp_price is not None and tp_price >= Decimal("101.30")
        assert tp_price > position.leg1_entry_price

        target_mid = tp_price + Decimal("0.05")
        for _ in range(3):
            await _tick(engine, clock, gateway, target_mid, Decimal("104.00"))
            if engine.ledger.trade_count:
                break
        assert engine.ledger.trade_count == 1, f"состояние={engine.state}"
        trade = engine.ledger.trades[0]
        assert trade.exit_reason is ExitReason.TAKE_PROFIT
        assert trade.net_pnl_rub > 0
        assert trade.leg1_exit_price == tp_price
        assert gateway.net_position(1) == 0

    asyncio.run(scenario())


def test_ghost_reversion_uses_scratch_limit_not_market_dump(tmp_path):
    """Правило 3: реверсия только по Leg2 не приводит к рыночному дампу Leg1."""

    async def scenario():
        config, clock, fee, gateway, engine = _bundle(
            tmp_path,
            ExecutionTopology.SINGLE_LEG_LEAD_LAG,
            _params(ghost_reversion_z=0.6, ghost_flat_seconds=45, scratch_seconds=60),
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)

        for _ in range(46):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("101.00"))
        assert engine.ledger.trade_count == 0, "не должно быть рыночного закрытия"
        assert engine.position is not None
        scratch_id = engine._roles.get(ROLE_TP_LEG1)  # noqa: SLF001
        assert scratch_id is not None
        scratch_price = gateway.order(scratch_id).price
        assert scratch_price == Decimal("100.01"), scratch_price
        assert engine.position.tags.get("scratch") is True

        for _ in range(140):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("101.00"))
            if engine.ledger.trade_count:
                break
        assert engine.ledger.trade_count == 1
        trade = engine.ledger.trades[0]
        assert trade.exit_reason is ExitReason.SCRATCH
        assert trade.net_pnl_rub <= 0

    asyncio.run(scenario())


def test_profit_floor_blocks_z_take_profit(tmp_path):
    """Правило 1: Z-выход запрещён, пока не покрыты издержки."""

    async def scenario():
        config, clock, fee, gateway, engine = _bundle(
            tmp_path,
            ExecutionTopology.SINGLE_LEG_LEAD_LAG,
            _params(ghost_flat_seconds=99999),
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        # Возвращаем спред точно к замороженному среднему: z_frozen ≈ 0, но Leg1 не сдвинулся.
        target_mid2 = Decimal("100.00") + Decimal(repr(round(position.frozen_mu, 2)))
        for _ in range(10):
            await _tick(engine, clock, gateway, Decimal("100.00"), target_mid2)
        assert engine.ledger.trade_count == 0, "тейк-профит не должен сработать без прибыли"
        assert engine.position is not None
        assert engine.state is FsmState.IN_POSITION

    asyncio.run(scenario())


def test_frozen_mu_prevents_drifting_target(tmp_path):
    """Правило 2: удержание у девиации не «схлопывает» Z из-за дрейфа среднего."""

    async def scenario():
        config, clock, fee, gateway, engine = _bundle(
            tmp_path,
            ExecutionTopology.SINGLE_LEG_LEAD_LAG,
            _params(ghost_flat_seconds=99999, max_hold_seconds=600),
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        for _ in range(60):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"))
        assert engine.ledger.trade_count == 0
        assert engine.position is not None
        frozen_z = engine._z_for_decision(engine.position, engine.metrics)  # noqa: SLF001
        assert abs(frozen_z) > 0.2
        assert abs(engine.metrics.zscore) < abs(frozen_z)

    asyncio.run(scenario())


def test_v1_mode_round_trip_is_still_supported(tmp_path):
    """Совместимость: с выключенными v2-фичами работает прежний Z-выход."""

    async def scenario():
        params = _params(
            use_frozen_mu=False,
            use_profit_lock=False,
            beta_mode="static",
            max_hold_seconds=600,
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _warmup(engine, clock, gateway)
        for _ in range(6):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"))
        assert engine.position is not None
        for _ in range(20):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("101.00"))
        await engine.shutdown(flatten=True)
        return engine

    engine = asyncio.run(scenario())
    assert engine.ledger.trade_count == 1


def test_dual_leg_v2_closes_when_spread_converges(tmp_path):
    async def scenario():
        config, clock, fee, gateway, engine = _bundle(
            tmp_path,
            ExecutionTopology.DUAL_LEG_HEDGED,
            _params(ghost_flat_seconds=99999),
        )
        await gateway.start()
        await _warmup(engine, clock, gateway)
        for _ in range(3):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"))
        assert engine.position is not None
        assert engine.position.leg2_filled_qty >= 1
        position = engine.position
        target_mid2 = Decimal("100.00") + Decimal(repr(round(position.frozen_mu, 2)))
        for _ in range(30):
            await _tick(engine, clock, gateway, Decimal("100.00"), target_mid2)
            if engine.ledger.trade_count:
                break
        await engine.shutdown(flatten=True)
        return engine, gateway

    engine, gateway = asyncio.run(scenario())
    assert engine.ledger.trade_count == 1, f"состояние={engine.state}"
    trade = engine.ledger.trades[0]
    assert trade.topology is ExecutionTopology.DUAL_LEG_HEDGED
    assert trade.leg2_entry_price is not None and trade.leg2_exit_price is not None
    assert gateway.net_position(1) == 0 and gateway.net_position(2) == 0


def test_time_stop_quotes_passively_before_market(tmp_path):
    """v2.1: тайм-стоп сначала выставляет мейкерский лимит, рынок — только через 30 с."""

    async def scenario():
        config, clock, fee, gateway, engine = _bundle(
            tmp_path,
            ExecutionTopology.SINGLE_LEG_LEAD_LAG,
            _params(
                max_hold_seconds=1,
                time_stop_quote_seconds=30,
                ghost_flat_seconds=99999,
                max_exit_quote_seconds=5,
            ),
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)

        await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"))
        assert engine.ledger.trade_count == 0, "нельзя закрывать позицию рынком сразу"
        assert engine.state is FsmState.QUOTING_EXIT_LEG1, engine.state
        exit_id = engine._roles.get(ROLE_EXIT_LEG1)  # noqa: SLF001
        assert exit_id is not None
        order = gateway.order(exit_id)
        assert order is not None and order.price == Decimal("100.01")

        for _ in range(34):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"))
            if engine.ledger.trade_count:
                break
        assert engine.ledger.trade_count == 1, f"состояние={engine.state}"
        assert engine.ledger.trades[0].exit_reason is ExitReason.TIME_STOP

    asyncio.run(scenario())


def test_warmup_requires_enough_ticks(tmp_path):
    async def scenario():
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, _params(ewma_window=500)
        )
        await gateway.start()
        for _ in range(20):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("101.00"))
        return engine

    engine = asyncio.run(scenario())
    assert engine.state is FsmState.WARMUP
    assert engine.metrics is not None and engine.metrics.is_warm is False


# ---------------------------------------------------------------------- v2.2
def test_z_peak_hook_thresholds(tmp_path):
    """Хук пика: вход только после отката |Z| от локального максимума."""
    config, clock, fee, gateway, engine = _bundle(
        tmp_path,
        ExecutionTopology.SINGLE_LEG_LEAD_LAG,
        _params(use_z_peak_hook=True, z_entry=2.8, z_hook_pullback=0.12, z_peak_reset_gap=0.25),
    )
    assert engine._z_peak_confirmed(2.90) is False  # noqa: SLF001 — пика ещё не было
    engine._z_peak = 2.95  # noqa: SLF001
    assert engine._z_peak_confirmed(2.95) is False  # noqa: SLF001 — импульс продолжается
    assert engine._z_peak_confirmed(2.83) is True  # noqa: SLF001 — откат 0.12
    assert engine._z_peak_confirmed(2.90) is False  # noqa: SLF001 — откат всего 0.05

    engine._z_peak = 2.95  # noqa: SLF001
    engine._z_pullback_streak = 2  # noqa: SLF001
    assert engine._z_peak_confirmed(2.80) is True  # noqa: SLF001 — два тика против Z

    engine._update_z_peak(0.5)  # noqa: SLF001 — ниже Z_entry - 0.25
    assert engine._z_peak == 0.0  # noqa: SLF001

    engine._update_z_peak(3.10)  # noqa: SLF001
    assert engine._z_peak == 3.10  # noqa: SLF001


def test_z_peak_hook_blocks_entry_while_spread_expands(tmp_path):
    """Пока |Z| расширяется, вход не открывается; после отката — открывается."""

    async def scenario():
        params = _params(
            z_entry=1.0,
            z_stop=1e9,
            ewma_window=100,
            use_entry_z_cap=False,
            use_z_peak_hook=True,
            z_hook_pullback=0.12,
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _warmup(engine, clock, gateway, ticks=120)
        peak = 0.0
        for step in range(1, 12):
            mid2 = Decimal("101.00") + Decimal("0.02") * step
            await _tick(engine, clock, gateway, Decimal("100.00"), mid2)
            peak = max(peak, abs(engine.metrics.zscore))
            if engine.position is not None:
                break
        assert engine.counters["hookSkips"] > 0
        assert engine.position is not None, f"вход не открылся: {engine.state}"
        assert engine.position.entry_z < peak, (engine.position.entry_z, peak)

    asyncio.run(scenario())


def test_breakeven_ratchet_arms_only_on_deep_mfe(tmp_path):
    """v2.3: взвод только от max(2.2·Fee, 0.90·цель) + 1 шаг, Z-триггер убран."""

    async def scenario():
        params = _params(ghost_flat_seconds=99999, z_stop=1e9, use_breakeven_ratchet=True)
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        fee_rt = position.round_trip_fee_rub
        step_rub = engine._step_value_rub(position)  # noqa: SLF001
        target = position.tp_target_rub
        arm_threshold = max(
            fee_rt * params.ratchet_arm_multiplier,
            target * Decimal(str(params.ratchet_arm_target_fraction)),
        ) + step_rub * params.ratchet_arm_step_buffer

        # Небольшой пик прибыли (было достаточно для v2.2) больше не взводит ратчет.
        position.max_favorable_gross_rub = Decimal("0.30")
        await _tick(engine, clock, gateway, Decimal("100.30"), Decimal("104.00"))
        assert position.breakeven_locked is False

        # Пик выше калиброванного порога взводит ратчет.
        position.max_favorable_gross_rub = arm_threshold + Decimal("0.01")
        await _tick(engine, clock, gateway, Decimal("100.30"), Decimal("104.00"))
        assert position.breakeven_locked is True
        assert engine.state is FsmState.IN_POSITION

    asyncio.run(scenario())


def test_breakeven_ratchet_does_not_abort_converging_trade(tmp_path):
    """Сходимость спреда к нулю (|z_frozen| <= 0.5) больше НЕ прерывает сделку."""

    async def scenario():
        params = _params(ghost_flat_seconds=99999, z_stop=1e9, use_breakeven_ratchet=True)
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        position.max_favorable_gross_rub = Decimal("5.00")
        position.breakeven_locked = True
        clock.advance(180_000)  # удержание > 120 с

        metrics = engine.metrics
        # z_frozen = 0 (полная сходимость), прибыль не откатилась -> выхода быть не должно.
        exited = await engine._manage_breakeven_ratchet(  # noqa: SLF001
            position, metrics, 0.0, Decimal("5.00")
        )
        assert exited is False
        assert engine.state is FsmState.IN_POSITION
        # А при реальном провале на 2+ шага и прибыли ниже 1.15·Fee — выход разрешён.
        exited = await engine._manage_breakeven_ratchet(  # noqa: SLF001
            position, metrics, 0.0, Decimal("0.05")
        )
        assert exited is True
        assert engine.state is FsmState.QUOTING_EXIT_LEG1

    asyncio.run(scenario())


def test_breakeven_ratchet_exits_after_hold_and_two_step_retrace(tmp_path):
    """Выход только после 120 с и отката на 2 шага ниже MFE к уровню 1.15·Fee."""

    async def scenario():
        params = _params(ghost_flat_seconds=99999, z_stop=1e9, use_breakeven_ratchet=True)
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        fee_rt = position.round_trip_fee_rub
        position.max_favorable_gross_rub = Decimal("3.00")
        position.breakeven_locked = True

        # Откат уже случился, но удержание < 120 с — выхода нет.
        await _tick(engine, clock, gateway, Decimal("100.08"), Decimal("104.00"))
        assert engine.state is FsmState.IN_POSITION

        for _ in range(125):
            await _tick(engine, clock, gateway, Decimal("100.08"), Decimal("104.00"))
            if engine.state is not FsmState.IN_POSITION:
                break
        assert engine.state is FsmState.QUOTING_EXIT_LEG1, f"состояние={engine.state}"
        exit_id = engine._roles.get(ROLE_EXIT_LEG1)  # noqa: SLF001
        assert exit_id is not None
        exit_price = gateway.order(exit_id).price
        for _ in range(3):
            await _tick(engine, clock, gateway, exit_price + Decimal("0.05"), Decimal("104.00"))
            if engine.ledger.trade_count:
                break
        assert engine.ledger.trade_count == 1, f"состояние={engine.state}"
        trade = engine.ledger.trades[0]
        assert trade.exit_reason is ExitReason.BREAKEVEN
        assert trade.leg1_exit_price == exit_price
        assert trade.gross_pnl_rub > 0

    asyncio.run(scenario())


def test_breakeven_ratchet_skipped_for_dual_leg(tmp_path):
    """v2.2.1: по умолчанию ratchet не применяется к DualLegHedged."""

    async def scenario():
        params = _params(ghost_flat_seconds=99999, z_stop=1e9, use_breakeven_ratchet=True)
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.DUAL_LEG_HEDGED, params
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        position.max_favorable_gross_rub = Decimal("50.00")
        metrics = engine.metrics
        assert await engine._manage_breakeven_ratchet(  # noqa: SLF001
            position, metrics, 0.0, Decimal("0.00")
        ) is False
        assert position.breakeven_locked is False
        assert engine.state is not FsmState.QUOTING_EXIT_LEG1

    asyncio.run(scenario())


def test_breakeven_ratchet_disabled_keeps_position(tmp_path):
    """Без ratchet-фиксации та же просадка не закрывает позицию."""

    async def scenario():
        params = _params(ghost_flat_seconds=99999, z_stop=1e9, use_breakeven_ratchet=False)
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        await _tick(engine, clock, gateway, Decimal("100.20"), Decimal("104.00"))
        await _tick(engine, clock, gateway, Decimal("100.08"), Decimal("104.00"))
        assert engine.state is FsmState.IN_POSITION
        assert engine.position is not None and engine.position.breakeven_locked is False

    asyncio.run(scenario())


def test_tp_target_decays_with_holding_time(tmp_path):
    """После половины MaxHold цель тейка сжимается к 1.15·Fee."""
    config, clock, fee, gateway, engine = _bundle(
        tmp_path,
        ExecutionTopology.SINGLE_LEG_LEAD_LAG,
        _params(use_tp_decay=True, max_hold_seconds=600, tp_decay_start_fraction=0.5),
    )

    async def scenario():
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        opened = position.opened_ms
        fee_rt = position.round_trip_fee_rub
        early = engine._target_tp_rub(position, opened + 60_000)  # noqa: SLF001
        late = engine._target_tp_rub(position, opened + 600_000)  # noqa: SLF001
        assert late < early, (early, late)
        assert late >= fee_rt * Decimal("1.15") - fee_rt * Decimal("0.01")
        assert engine._tp_price_for(position, late) < engine._tp_price_for(position, early)  # noqa: SLF001

    asyncio.run(scenario())


def test_regime_filter_directions(tmp_path):
    """Кейс A (ведёт нога 2), кейс B (выброс ноги 1) и блокировка тренда."""
    config, clock, fee, gateway, engine = _bundle(
        tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, _params(use_regime_filter=True)
    )

    def fill(delta1: float, delta2: float) -> None:
        engine._regime_v1.clear()  # noqa: SLF001
        engine._regime_v2.clear()  # noqa: SLF001
        for index in range(13):
            engine._regime_v1.append(100.0 + delta1 * index / 12.0)  # noqa: SLF001
            engine._regime_v2.append(200.0 + delta2 * index / 12.0)  # noqa: SLF001

    fill(0.10, 2.00)
    assert engine._regime_confirmed(2.5) is True  # noqa: SLF001 — кейс A
    fill(0.10, -2.00)
    assert engine._regime_confirmed(2.5) is False  # noqa: SLF001 — нога 2 против сигнала
    fill(-2.00, 0.10)
    assert engine._regime_confirmed(2.5) is True  # noqa: SLF001 — кейс B (фейд выброса)
    fill(-2.00, 0.10)
    assert engine._regime_confirmed(-2.5) is False  # noqa: SLF001 — выброс в сторону сигнала
    fill(1.50, 1.60)
    assert engine._regime_confirmed(2.5) is False  # noqa: SLF001 — общий тренд, вход блокируем
    assert engine._regime_confirmed(2.5) is False  # noqa: SLF001

    dual_config, dual_clock, dual_fee, dual_gateway, dual_engine = _bundle(
        tmp_path, ExecutionTopology.DUAL_LEG_HEDGED, _params(use_regime_filter=True)
    )
    fill(1.50, 1.60)
    assert dual_engine._regime_confirmed(2.5) is True  # noqa: SLF001 — фильтр только для SingleLeg
