"""Тесты v2.3: ценовой стоп SingleLeg, выпуклость тейка и внутридневные окна."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from bot_a.models.domain import ExecutionTopology, FsmState
from bot_a.risk.guard import IntradayEntryWindows
from bot_a.utils.time_provider import MSK
from tests.test_engine import (
    ROLE_TP_LEG1,
    _bundle,
    _open_position,
    _params,
    _tick,
    _warmup,
)

import asyncio
import pytest


def test_session_window_modes():
    all_day = IntradayEntryWindows("AllDay")
    main = IntradayEntryWindows("MainSessionOnly")
    prime = IntradayEntryWindows("PrimeHoursOnly")

    assert all_day.allows(datetime(2026, 10, 20, 23, 0, tzinfo=MSK)) is True
    assert main.allows(datetime(2026, 10, 20, 23, 0, tzinfo=MSK)) is False
    assert "основной сессии" in (main.blocking_reason(datetime(2026, 10, 20, 23, 0, tzinfo=MSK)) or "")

    # Обед (13:00) — только MainSessionOnly, «прайм» закрыт.
    lunch = datetime(2026, 10, 20, 13, 0, tzinfo=MSK)
    assert main.allows(lunch) is True
    assert prime.allows(lunch) is False
    assert "активных часов" in (prime.blocking_reason(lunch) or "")

    # Утро и вторая половина дня — открыты для всех режимов.
    for hour in (10, 11, 16, 18):
        moment = datetime(2026, 10, 20, hour, 10, tzinfo=MSK)
        assert all_day.allows(moment) is True
        assert main.allows(moment) is True
        assert prime.allows(moment) is True

    # 18:36 — вне основной сессии и вне «прайма», вечер (20:00) — только AllDay.
    assert main.allows(datetime(2026, 10, 20, 18, 40, tzinfo=MSK)) is False
    assert prime.allows(datetime(2026, 10, 20, 8, 0, tzinfo=MSK)) is False
    assert IntradayEntryWindows("unknown-mode").mode == "AllDay"


def test_single_leg_no_false_hardstop_when_lead_extends(tmp_path):
    """v2.3: рост |Z_frozen| при неподвижной Leg1 НЕ должен вызывать HardStop."""

    async def scenario():
        params = _params(
            z_stop=3.0,
            z_entry=1.0,
            use_entry_z_cap=False,
            ghost_flat_seconds=99999,
            use_breakeven_ratchet=False,
            use_tp_decay=False,
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        # Leg1 стоит на месте, Leg2 уходит дальше в сторону сигнала -> |Z| растёт.
        for mid2 in ("106.00", "108.00", "110.00"):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal(mid2))
        assert engine.position is position, "позиция не должна закрываться"
        assert engine.state is FsmState.IN_POSITION, f"состояние={engine.state}"
        assert engine.ledger.trade_count == 0

    asyncio.run(scenario())


def test_single_leg_stop_on_leg1_mae(tmp_path):
    """v2.3: стоп по фактической просадке Leg1 (target × 1.5, не меньше 4 шагов)."""

    async def scenario():
        params = _params(
            z_stop=1e9,
            ghost_flat_seconds=99999,
            use_breakeven_ratchet=False,
            use_tp_decay=False,
            single_leg_stop_to_target_ratio=1.5,
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        target = position.tp_target_rub
        step_rub = engine._step_value_rub(position)  # noqa: SLF001
        stop_rub = max(target * Decimal("1.5"), step_rub * 4)
        loss_steps = int(stop_rub / step_rub) + 3
        entry = position.leg1_entry_price
        loss_price = entry - Decimal(loss_steps) * Decimal("0.01")
        await _tick(engine, clock, gateway, loss_price, Decimal("104.00"))
        for _ in range(4):
            if engine.ledger.trade_count:
                break
            await _tick(engine, clock, gateway, loss_price, Decimal("104.00"))
        assert engine.ledger.trade_count == 1, f"состояние={engine.state}"
        trade = engine.ledger.trades[0]
        assert trade.exit_reason.value == "HardStop"
        assert trade.gross_pnl_rub < 0

    asyncio.run(scenario())


def test_single_leg_lead_failure_stop(tmp_path):
    """Стоп по развороту ведущей ноги требует убытка Leg1 и выдержки leadFailureSeconds."""

    async def scenario():
        params = _params(
            ghost_flat_seconds=99999,
            use_breakeven_ratchet=False,
            use_tp_decay=False,
            lead_failure_z=0.5,
            lead_failure_seconds=60,
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        await _tick(engine, clock, gateway, Decimal("99.97"), Decimal("104.00"))
        metrics = engine.metrics
        reason_short = engine._single_leg_stop_reason(position, metrics, -1.5, 10.0)  # noqa: SLF001
        assert reason_short is None, "слишком рано: выдержка не набрана"
        reason = engine._single_leg_stop_reason(position, metrics, -1.5, 61.0)  # noqa: SLF001
        assert reason is not None and "импульс Leg2" in reason
        # Без убытка Leg1 стоп не срабатывает даже при развороте импульса.
        await _tick(engine, clock, gateway, Decimal("100.05"), Decimal("104.00"))
        profit_reason = engine._single_leg_stop_reason(position, engine.metrics, -1.5, 61.0)  # noqa: SLF001
        assert profit_reason is None

    asyncio.run(scenario())


def test_dual_leg_keeps_z_stop(tmp_path):
    """DualLegHedged по-прежнему закрывается по |Z_frozen| >= Z_stop."""

    async def scenario():
        params = _params(
            z_stop=3.0,
            use_entry_z_cap=False,
            ghost_flat_seconds=99999,
            use_breakeven_ratchet=False,
            use_tp_decay=False,
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.DUAL_LEG_HEDGED, params
        )
        await gateway.start()
        await _warmup(engine, clock, gateway)
        for _ in range(2):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"))
        assert engine.position is not None
        for _ in range(4):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("107.00"))
            if engine.ledger.trade_count:
                break
        assert engine.ledger.trade_count == 1
        assert engine.ledger.trades[0].exit_reason.value == "HardStop"

    asyncio.run(scenario())


def test_tp_target_respects_fee_floor_and_deviation(tmp_path):
    """Цель тейка = max(2.5·Fee, 1.35·Fee, dev·fraction), без сжатия по времени."""

    async def scenario():
        params = _params(
            ghost_flat_seconds=99999,
            use_breakeven_ratchet=False,
            use_tp_decay=False,
            tp_min_fee_multiple=Decimal("2.5"),
            tp_deviation_fraction=0.85,
        )
        config, clock, fee, gateway, engine = _bundle(
            tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        fee_rt = position.round_trip_fee_rub
        target = engine._target_tp_rub(position, position.opened_ms + 5_000)  # noqa: SLF001
        late = engine._target_tp_rub(position, position.opened_ms + 599_000)  # noqa: SLF001
        assert target >= fee_rt * Decimal("2.5") - Decimal("0.001")
        assert target == late, "при useTpDecay=false цель не сжимается"
        tp_order = engine._roles.get(ROLE_TP_LEG1)  # noqa: SLF001
        assert tp_order is not None
        tp_price = gateway.order(tp_order).price
        expected_delta = target / engine._leg1.rub_per_price_unit
        actual_delta = abs(tp_price - position.leg1_entry_price)
        assert actual_delta >= expected_delta - Decimal("0.01")

    asyncio.run(scenario())


def test_entry_blocked_outside_active_window(tmp_path):
    """v2.3: вне активного окна новые входы не открываются (выходы — разрешены)."""

    async def scenario():
        from bot_a.engine.factory import create_engine
        from bot_a.engine.ledger import PortfolioLedger
        from bot_a.economics.fees import TinkoffFeeCalculator
        from bot_a.execution.simulated_gateway import SimulatedOrderExecutionGateway
        from bot_a.risk.guard import DailyKillSwitch, MoexSessionSchedule, RiskAndScheduleGuard
        from bot_a.utils.time_provider import SimulatedTimeProvider
        from tests.helpers import make_book, make_config, make_meta, make_preset, make_trade, trading_start_ms

        leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
        leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
        params = _params(ghost_flat_seconds=99999, use_breakeven_ratchet=False, use_tp_decay=False)
        preset = make_preset(leg1, leg2)
        config = make_config(tmp_path, preset, params, ExecutionTopology.SINGLE_LEG_LEAD_LAG)
        # 10:00 MSK — раньше начала активных окон (10:03 / 10:05).
        clock = SimulatedTimeProvider(trading_start_ms())
        fee = TinkoffFeeCalculator(config.fees.tariff, clock=clock)
        gateway = SimulatedOrderExecutionGateway(
            metadata=[leg1, leg2], clock=clock, fee_calculator=fee, latency_ms=0,
            queue_fill_factor=0.35, slippage_steps=0, order_book_depth=5,
            initial_cash_rub=config.execution.paper_initial_cash_rub,
        )
        guard = RiskAndScheduleGuard(
            MoexSessionSchedule(futures=True),
            DailyKillSwitch(config.risk.max_daily_drawdown_rub),
            config.risk.margin_safety_buffer_pct,
            config.risk.max_position_lots,
            IntradayEntryWindows("MainSessionOnly"),
        )
        ledger = PortfolioLedger(config.execution.paper_initial_cash_rub)
        engine = create_engine(
            [leg1, leg2], params, ExecutionTopology.SINGLE_LEG_LEAD_LAG,
            gateway, fee, clock, guard, ledger,
        )
        await gateway.start()
        await _warmup(engine, clock, gateway)
        for _ in range(3):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"))
        assert engine.position is None, "вход должен быть заблокирован окном"
        assert engine.counters["sessionSkips"] > 0
        assert engine.state is FsmState.IDLE

        # После 10:05 вход разрешён.
        clock.advance(6 * 60 * 1000)
        for _ in range(4):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"))
            if engine.position is not None:
                break
        assert engine.position is not None, f"вход не открылся: {engine.state}"

    asyncio.run(scenario())
