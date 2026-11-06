"""Тесты v4.0: мульти-режимный движок (пробой + откат + фейд)."""

from __future__ import annotations

import pytest

from bot_a.engine.breakout_fsm import (
    Bar,
    BreakoutExitReason,
    BreakoutParams,
    BreakoutSide,
    MomentumBreakoutStateMachine,
)
from bot_a.engine.multiregime_fsm import MultiRegimeStateMachine, regime_machine
from bot_a.utils.quotation import dec
from tests.helpers import make_meta
from tests.test_v32_swing import bar, minute_ms, v32_params


def multi_params(**overrides) -> BreakoutParams:
    base = v32_params(
        regime_mode="multi",
        volume_multiple=0.0,
        max_squeeze_ratio=0.0,
        min_atr_to_fee_ratio=0.0,
        close_in_range=0.3,
        pullback_enabled=True,
        fade_enabled=True,
        pullback_min_er=0.22,
        fade_max_er=0.20,
        pullback_stop_atr=1.5,
        pullback_trail_atr=3.2,
        # В тестах v4.0 проверяется именно правило касания EMA20; фильтр импульса
        # и волновой кулдаун (v4.1) тестируются отдельно в test_v41_multiregime.py.
        pullback_require_thrust=False,
        pullback_wave_cooldown_bars=0,
        fade_keltner_atr=2.2,
        fade_stop_atr=1.5,
        fade_min_target_fee_ratio=2.0,
        donchian_bars=10,
        # Реалистичные EMA тренда: EMA20 отстаёт от цены, поэтому «откат к EMA20»
        # действительно означает откат, а не любой бар в тренде.
        ema_fast_bars=20,
        ema_slow_bars=90,
    )
    return base.overrides(**overrides)


def machine(params: BreakoutParams) -> MultiRegimeStateMachine:
    return MultiRegimeStateMachine(
        make_meta("TEST", price_step="0.01", step_amount="0.01"), params
    )


def trend_bars(count: int = 120, step: float = 0.08) -> list[Bar]:
    """Плавный рост: EMA20 выше EMA90, ER высокий (нужно ≥ 90 баров для EMA90)."""
    return [
        bar(index, 100.0 + index * step, high=100.0 + index * step + 0.05, low=100.0 + index * step - 0.05)
        for index in range(count)
    ]


# ------------------------------------------------------------------ фабрика
def test_factory_returns_multiregime_only_for_multi() -> None:
    meta = make_meta("TEST")
    assert isinstance(regime_machine(meta, v32_params()), MomentumBreakoutStateMachine)
    assert not isinstance(regime_machine(meta, v32_params()), MultiRegimeStateMachine)
    assert isinstance(regime_machine(meta, multi_params()), MultiRegimeStateMachine)


def test_params_parse_v40_fields_from_camel_case() -> None:
    params = BreakoutParams.from_mapping(
        {
            "regimeMode": "multi",
            "pullbackEnabled": True,
            "pullbackMinEr": 0.3,
            "fadeEnabled": True,
            "fadeMaxEr": 0.15,
            "fadeKeltnerAtr": 2.5,
        }
    )
    assert params.regime_mode == "multi"
    assert params.pullback_enabled is True
    assert params.pullback_min_er == pytest.approx(0.3)
    assert params.fade_enabled is True
    assert params.fade_max_er == pytest.approx(0.15)
    assert params.fade_keltner_atr == pytest.approx(2.5)
    # По умолчанию движки B/C выключены — прежнее поведение сохраняется.
    assert BreakoutParams().pullback_enabled is False
    assert BreakoutParams().fade_enabled is False
    assert BreakoutParams().regime_mode == "breakout"


# ------------------------------------------------------------------ Engine B
def test_pullback_engine_enters_on_ema_touch() -> None:
    engine = machine(multi_params())
    for item in trend_bars():
        engine.on_bar(item)
    assert engine.position is None
    ema_fast = engine._indicators.snapshot.ema_fast  # noqa: SLF001
    # Бар дотягивается до EMA20, но тренд не сломан (закрытие выше средней).
    touch_bar = bar(120, ema_fast + 0.02, high=ema_fast + 0.10, low=ema_fast - 0.05)
    events = engine.on_bar(touch_bar)
    entries = [event for event in events if event.kind == "entry"]
    assert entries, "касание EMA20 в тренде должно давать вход"
    assert entries[0].side is BreakoutSide.LONG
    assert engine.counters.pullback_entries == 1
    position = engine.position
    assert position is not None
    assert position.style == "pullback"
    assert position.trail_mult == pytest.approx(3.2)
    # Стоп — за минимум отката.
    assert float(position.stop_price) < float(touch_bar.low)


def test_pullback_engine_requires_retrace() -> None:
    engine = machine(multi_params())
    for item in trend_bars():
        engine.on_bar(item)
    # Бар уходит вверх от максимумов, до EMA20 не дотягиваясь — это не откат.
    last = trend_bars()[-1]
    thrust = bar(120, float(last.high) + 0.2, high=float(last.high) + 0.25, low=float(last.high), open_=float(last.high))
    events = engine.on_bar(thrust)
    assert not [event for event in events if event.kind == "entry" and event.note.startswith("pullback")]
    assert engine.counters.pullback_entries == 0


def test_pullback_disabled_by_flag() -> None:
    engine = machine(multi_params(pullback_enabled=False, fade_enabled=False))
    for item in trend_bars():
        engine.on_bar(item)
    ema_fast = engine._indicators.snapshot.ema_fast  # noqa: SLF001
    engine.on_bar(bar(120, ema_fast + 0.02, high=ema_fast + 0.10, low=ema_fast - 0.05))
    assert engine.counters.pullback_entries == 0


# ------------------------------------------------------------------ Engine C
def chop_bars(count: int = 24) -> list[Bar]:
    """Боковик ±0.3 вокруг 100: ER низкий, EMA20 ≈ EMA90."""
    return [
        bar(index, 100.0 + (0.3 if index % 2 else -0.3), high=100.4, low=99.6)
        for index in range(count)
    ]


def fade_engine(**overrides) -> MultiRegimeStateMachine:
    engine = machine(multi_params(pullback_enabled=False, **overrides))
    for item in chop_bars():
        engine.on_bar(item)
    return engine


def test_fade_engine_fades_rejection_above_keltner() -> None:
    engine = fade_engine()
    snapshot = engine._indicators.snapshot  # noqa: SLF001
    assert snapshot.efficiency_ratio <= 0.20
    center = snapshot.ema_fast
    atr = snapshot.atr_points
    upper = center + 2.2 * atr
    # Бар выносит выше полосы, но закрывается обратно внутрь (в нижней части диапазона).
    spike = bar(120, upper - 0.55, high=upper + 0.45, low=upper - 0.6, open_=upper - 0.2)
    events = engine.on_bar(spike)
    entries = [event for event in events if event.kind == "entry"]
    assert entries, "отбой от верхней полосы должен давать шорт"
    assert entries[0].side is BreakoutSide.SHORT
    assert engine.counters.fade_entries == 1
    position = engine.position
    assert position is not None
    assert position.style == "fade"
    assert float(position.stop_price) > float(spike.high)
    assert float(position.target_price) == pytest.approx(center, abs=0.01)


def test_fade_target_exit_at_ema20() -> None:
    engine = fade_engine()
    snapshot = engine._indicators.snapshot  # noqa: SLF001
    center = snapshot.ema_fast
    atr = snapshot.atr_points
    upper = center + 2.2 * atr
    engine.on_bar(bar(120, upper - 0.55, high=upper + 0.45, low=upper - 0.6, open_=upper - 0.2))
    assert engine.position is not None
    # Возврат к EMA20 — выход по цели.
    events = engine.on_bar(bar(121, center + 0.01, high=upper - 0.2, low=center - 0.05))
    exits = [event for event in events if event.kind == "exit"]
    assert exits and exits[0].reason is BreakoutExitReason.TAKE_PROFIT
    assert engine.position is None


def test_fade_requires_fee_headroom() -> None:
    engine = fade_engine(fade_min_target_fee_ratio=50.0)
    snapshot = engine._indicators.snapshot  # noqa: SLF001
    center = snapshot.ema_fast
    atr = snapshot.atr_points
    upper = center + 2.2 * atr
    events = engine.on_bar(bar(120, upper - 0.55, high=upper + 0.45, low=upper - 0.6, open_=upper - 0.2))
    assert not [event for event in events if event.kind == "entry"]
    assert engine.counters.fade_entries == 0


def test_fade_stop_caps_loss() -> None:
    engine = fade_engine()
    snapshot = engine._indicators.snapshot  # noqa: SLF001
    center = snapshot.ema_fast
    atr = snapshot.atr_points
    upper = center + 2.2 * atr
    engine.on_bar(bar(120, upper - 0.55, high=upper + 0.45, low=upper - 0.6, open_=upper - 0.2))
    position = engine.position
    assert position is not None
    stop = float(position.stop_price)
    events = engine.on_bar(bar(121, stop + 0.2, high=stop + 0.3, low=stop - 0.1))
    exits = [event for event in events if event.kind == "exit"]
    assert exits and exits[0].reason is BreakoutExitReason.HARD_STOP


# ------------------------------------------------------------------ маршрутизация режимов
def test_trend_regime_blocks_fade_engine() -> None:
    engine = machine(multi_params(pullback_enabled=False))
    for item in trend_bars():
        engine.on_bar(item)
    snapshot = engine._indicators.snapshot  # noqa: SLF001
    assert snapshot.efficiency_ratio > 0.20  # тренд — фейд не должен включаться
    center = snapshot.ema_fast
    atr = snapshot.atr_points
    upper = center + 2.2 * atr
    events = engine.on_bar(bar(120, upper - 0.55, high=upper + 0.45, low=upper - 0.6, open_=upper - 0.2))
    assert not [event for event in events if event.kind == "entry" and event.note.startswith("fade")]


def test_breakout_engine_still_works_in_multi_mode() -> None:
    engine = machine(multi_params(pullback_enabled=False, fade_enabled=False))
    from tests.test_v32_swing import _breakout_bars

    events = []
    for item in _breakout_bars():
        events.extend(engine.on_bar(item))
    assert [event for event in events if event.kind == "entry"]


def test_regime_counters_reported() -> None:
    engine = fade_engine()
    snapshot = engine._indicators.snapshot  # noqa: SLF001
    upper = snapshot.ema_fast + 2.2 * snapshot.atr_points
    engine.on_bar(bar(120, upper - 0.55, high=upper + 0.45, low=upper - 0.6, open_=upper - 0.2))
    payload = engine.counters.as_dict()
    assert payload["fadeEntries"] == 1
    assert "pullbackEntries" in payload
