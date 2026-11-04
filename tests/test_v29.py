"""Тесты v2.9: потолок тейка только для β < 0.8, выход из «мёртвого» спреда.

Обе механики выросли из 5-дневного бенчмарка реальных L2 (`28.10–02.11.2026`):

1. потолок тейка (`2.2σ` / `0.45 %` номинала) вводился ради `AFKS` (β = 0.377), но
   для `VTBR` (β ≈ 5.1) и `MTLR` (β ≈ 7.6) он резал прибыльные тейки вдвое —
   `MTLR` на реальных L2 давал `+11.23 RUB` вместо `+22.04 RUB`;
2. единственный убыток портфеля — `IRAO_SBER` 02.11 (`BUY 11:22:41 → 11:53:11`,
   `−15.91 RUB`): 12 минут спред стоял, но выход «по мёртвому спреду» без защиты
   от просадки закрыл его ещё хуже (`−22.71 RUB`).
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal

from bot_a.models.domain import ExecutionTopology, ExitReason
from tests.test_engine import _bundle, _open_position, _params

SINGLE = ExecutionTopology.SINGLE_LEG_LEAD_LAG


def _capped_position(tmp_path, beta: float, **overrides):
    """Открывает позицию с заданной бетой и возвращает цель/потолок тейка."""

    async def scenario():
        base = {
            "tp_max_sigma_multiple": 2.2,
            "tp_max_notional_fraction": 0.0045,
            "tp_cap_beta_threshold": 0.8,
        }
        base.update(overrides)
        params = _params(**base)
        config, clock, _fee, gateway, engine = _bundle(
            tmp_path / f"b{beta}", SINGLE, params
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        assert position is not None
        position.frozen_beta = beta
        target = engine._target_tp_rub(position, clock.now_ms())  # noqa: SLF001
        cap = engine._tp_cap_rub(position)  # noqa: SLF001
        return target, cap

    return asyncio.run(scenario())


def test_tp_cap_skipped_for_beta_above_threshold(tmp_path):
    """β ≥ 0.8 (VTBR, MTLR, IRAO) — потолок тейка не применяется (v2.9)."""
    target, cap = _capped_position(tmp_path, 5.1)
    assert cap == Decimal("0"), "для β ≥ 0.8 потолок не должен считаться"
    assert target > 0


def test_tp_cap_applied_for_low_beta(tmp_path):
    """β < 0.8 (AFKS с β = 0.377) — потолок остаётся, иначе тейк уезжает на +10 %."""
    target, cap = _capped_position(tmp_path, 0.377)
    assert cap > 0, "для β < 0.8 потолок обязателен"
    assert target <= cap


def test_tp_cap_threshold_disabled_caps_every_pair(tmp_path):
    """`tpCapBetaThreshold = 0` — прежнее поведение v2.8 (потолок для всех пар)."""
    target, cap = _capped_position(tmp_path, 5.1, tp_cap_beta_threshold=0.0)
    assert cap > 0
    assert target <= cap


def _stagnation_engine(tmp_path, **overrides):
    base = {
        "use_stagnation_exit": True,
        "stagnation_after_seconds": 60,
        "stagnation_mfe_fee_multiple": 1.0,
        "stagnation_loss_fee_multiple": 1.5,
        "stagnation_min_z_reversion": 0.25,
    }
    base.update(overrides)
    return _bundle(tmp_path, SINGLE, _params(**base))


def _adverse_metrics(engine, position, loss_rub: Decimal):
    """Метрики, при которых позиция теряет ``loss_rub`` рублей (по Leg1)."""
    sign = Decimal(position.side_leg1.sign)
    price = position.leg1_entry_price - loss_rub * sign / engine._leg1.rub_per_price_unit  # noqa: SLF001
    return replace(engine.metrics, micro_leg1=price)


def _force_hold(position, hold_seconds: int, mfe: Decimal, mae: Decimal) -> None:
    """Сдвигает время открытия и фиксирует историю MFE/MAE до проверки триггера."""
    position.opened_ms -= hold_seconds * 1000
    position.max_favorable_gross_rub = mfe
    position.min_adverse_gross_rub = mae


def _stagnation_reason(engine, position, metrics, hold_seconds: float):
    return engine._stagnation_reason(  # noqa: SLF001
        position, metrics, engine._entry_z(position), hold_seconds  # noqa: SLF001
    )


def test_stagnation_exit_fires_when_spread_is_dead(tmp_path):
    """Спред стоит 12+ минут, MFE не дошёл до комиссии, |Z| не вернулся — выходим."""

    async def scenario():
        config, clock, _fee, gateway, engine = _stagnation_engine(tmp_path)
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        assert position is not None
        engine._freeze_position(position, engine.metrics)  # noqa: SLF001
        fee = position.round_trip_fee_rub
        assert fee > 0
        metrics = _adverse_metrics(engine, position, fee * 3)
        _force_hold(position, 120, mfe=fee * Decimal("0.2"), mae=-fee * 20)
        reason = _stagnation_reason(engine, position, metrics, 120.0)
        assert reason is not None, "мёртвый спред обязан закрыться"
        assert "удержание" in reason and "реверсия" in reason

    asyncio.run(scenario())


def test_stagnation_exit_waits_for_the_minimum_hold(tmp_path):
    """До `stagnationAfterSeconds` тайм-выход не срабатывает — это не стоп."""

    async def scenario():
        config, clock, _fee, gateway, engine = _stagnation_engine(tmp_path)
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        assert position is not None
        engine._freeze_position(position, engine.metrics)  # noqa: SLF001
        fee = position.round_trip_fee_rub
        metrics = _adverse_metrics(engine, position, fee * 3)
        _force_hold(position, 120, mfe=fee * Decimal("0.2"), mae=-fee * 20)
        assert _stagnation_reason(engine, position, metrics, 30.0) is None

    asyncio.run(scenario())


def test_stagnation_exit_skips_reverted_spread(tmp_path):
    """Если |Z| уже вернулся к нулю на 25 %+, сделка не «мёртвая» — держим."""

    async def scenario():
        config, clock, _fee, gateway, engine = _stagnation_engine(tmp_path)
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        assert position is not None
        engine._freeze_position(position, engine.metrics)  # noqa: SLF001
        fee = position.round_trip_fee_rub
        metrics = _adverse_metrics(engine, position, fee * 3)
        _force_hold(position, 120, mfe=fee * Decimal("0.2"), mae=-fee * 20)
        entry_z = engine._entry_z(position)  # noqa: SLF001
        assert entry_z > 1.0
        reverted_z = entry_z * 0.5  # вернулись на 50 % — больше порога 25 %
        assert (
            engine._stagnation_reason(position, metrics, reverted_z, 120.0)  # noqa: SLF001
            is None
        )

    asyncio.run(scenario())


def test_stagnation_exit_waits_if_mfe_reached_fee(tmp_path):
    """Цель уже была достижима (MFE > комиссии) — это не мёртвый спред."""

    async def scenario():
        config, clock, _fee, gateway, engine = _stagnation_engine(tmp_path)
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        assert position is not None
        engine._freeze_position(position, engine.metrics)  # noqa: SLF001
        fee = position.round_trip_fee_rub
        metrics = _adverse_metrics(engine, position, fee * 3)
        _force_hold(position, 120, mfe=fee * 2, mae=-fee * 20)
        assert _stagnation_reason(engine, position, metrics, 120.0) is None

    asyncio.run(scenario())


def test_stagnation_mae_guard_blocks_exit_at_worst_level(tmp_path):
    """IRAO 02.11: выход срабатывал ровно на максимуме просадки — защита по MAE."""

    async def scenario():
        config, clock, _fee, gateway, engine = _stagnation_engine(
            tmp_path, stagnation_max_mae_fraction=0.7
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        assert position is not None
        engine._freeze_position(position, engine.metrics)  # noqa: SLF001
        fee = position.round_trip_fee_rub
        loss = fee * 3
        metrics = _adverse_metrics(engine, position, loss)
        _force_hold(position, 120, mfe=fee * Decimal("0.2"), mae=-loss)
        # Текущий PnL равен MAE — позиция у худшего уровня, выхода нет.
        assert _stagnation_reason(engine, position, metrics, 120.0) is None
        # Восстановились до половины просадки — «мёртвый» выход разрешён.
        position.min_adverse_gross_rub = -loss * 2
        assert _stagnation_reason(engine, position, metrics, 120.0) is not None

    asyncio.run(scenario())


def test_stagnation_exit_runs_as_passive_time_exit(tmp_path):
    """Сработавший триггер закрывает позицию пассивным лимитом, не рынком."""

    async def scenario():
        config, clock, _fee, gateway, engine = _stagnation_engine(
            tmp_path, max_exit_quote_seconds=60
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        assert position is not None
        engine._freeze_position(position, engine.metrics)  # noqa: SLF001
        fee = position.round_trip_fee_rub
        hold = 120
        position.opened_ms -= hold * 1000
        metrics = _adverse_metrics(engine, position, fee * 3)
        await engine._manage_position(metrics)  # noqa: SLF001
        assert engine.state.name == "QUOTING_EXIT_LEG1", engine.state
        assert engine._exit_reason is ExitReason.STAGNATION  # noqa: SLF001
        assert engine.ledger.trade_count == 0, "рынком не закрываем"

    asyncio.run(scenario())
