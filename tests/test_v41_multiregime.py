"""Тесты v4.1: анти-churn откатов, дедупликация активов, сторож здоровья,
торможение риска на просадке и резолв живого фронт-контракта."""

from __future__ import annotations

from datetime import date

import pytest

from bot_a.backtest.breakout_backtest import BreakoutBacktester, BreakoutSettings
from bot_a.backtest.swing_portfolio import SwingAccountBacktester, SwingAccountSettings
from bot_a.engine.breakout_fsm import BreakoutParams, BreakoutSide
from bot_a.engine.multiregime_fsm import MultiRegimeStateMachine
from bot_a.services.front_contracts import (
    deduplicate,
    needs_resolution,
    resolve_front_contract,
    resolve_swing_tickers,
    underlying_family,
)
from tests.helpers import make_meta
from tests.test_v32_swing import bar, v32_params
from tests.test_v40_multiregime import chop_bars, machine, multi_params, trend_bars


# ------------------------------------------------------------------ Engine B: импульс
def test_pullback_requires_thrust_by_default() -> None:
    params = multi_params(pullback_require_thrust=True, pullback_wave_cooldown_bars=0)
    assert params.pullback_require_thrust is True
    engine = machine(params)
    for item in trend_bars():
        engine.on_bar(item)
    ema_fast = engine._indicators.snapshot.ema_fast  # noqa: SLF001
    # Касание без импульса: закрытие в середине бара, тело ~0 → вход запрещён.
    flat_touch = bar(120, ema_fast + 0.02, high=ema_fast + 0.05, low=ema_fast - 0.05, open_=ema_fast + 0.01)
    events = engine.on_bar(flat_touch)
    assert not [event for event in events if event.kind == "entry"]
    assert engine.counters.thrust_skips >= 1
    assert engine.counters.pullback_entries == 0


def test_pullback_enters_when_thrust_breaks_previous_high() -> None:
    params = multi_params(pullback_require_thrust=True, pullback_wave_cooldown_bars=0)
    engine = machine(params)
    for item in trend_bars():
        engine.on_bar(item)
    ema_fast = engine._indicators.snapshot.ema_fast  # noqa: SLF001
    previous = engine._prev_bar  # noqa: SLF001
    assert previous is not None
    thrust = bar(
        120,
        float(previous.high) + 0.05,
        high=float(previous.high) + 0.08,
        low=ema_fast - 0.03,
        open_=ema_fast + 0.01,
    )
    events = engine.on_bar(thrust)
    entries = [event for event in events if event.kind == "entry"]
    assert entries and entries[0].side is BreakoutSide.LONG
    assert engine.counters.pullback_entries == 1
    assert engine.counters.thrust_skips == 0


def test_thrust_can_come_from_strong_body() -> None:
    params = multi_params(
        pullback_require_thrust=True,
        pullback_thrust_min_atr=0.25,
        pullback_wave_cooldown_bars=0,
    )
    engine = machine(params)
    for item in trend_bars():
        engine.on_bar(item)
    ema_fast = engine._indicators.snapshot.ema_fast  # noqa: SLF001
    atr = engine._indicators.snapshot.atr_points  # noqa: SLF001
    previous = engine._prev_bar  # noqa: SLF001
    # Максимум предыдущего бара НЕ пробит, но тело ≥ 0.25 ATR, закрытие в верхних 35 %,
    # а минимум доктягивается до EMA20.
    close = ema_fast + 0.12
    open_ = close - 0.8 * atr  # тело заведомо больше порога (ATR пересчитывается с этим баром)
    low = ema_fast - 0.03
    strong = bar(120, close, high=close + 0.02, low=low, open_=open_)
    assert float(close) < float(previous.high)
    assert strong.close_position >= 0.65
    events = engine.on_bar(strong)
    assert [event for event in events if event.kind == "entry"]
    assert engine.counters.thrust_skips == 0


# ------------------------------------------------------------------ Engine B: кулдаун
def test_pullback_wave_cooldown_blocks_second_entry() -> None:
    params = multi_params(pullback_require_thrust=False, pullback_wave_cooldown_bars=6)
    engine = machine(params)
    for item in trend_bars():
        engine.on_bar(item)
    ema_fast = engine._indicators.snapshot.ema_fast  # noqa: SLF001
    first = bar(120, ema_fast + 0.02, high=ema_fast + 0.06, low=ema_fast - 0.05)
    events = engine.on_bar(first)
    assert [event for event in events if event.kind == "entry"]
    engine.position = None  # имитируем выход: стоп сработал на следующем баре
    second = bar(121, ema_fast + 0.03, high=ema_fast + 0.07, low=ema_fast - 0.04)
    events = engine.on_bar(second)
    assert not [event for event in events if event.kind == "entry"]
    assert engine.counters.wave_skips >= 1


def test_pullback_cooldown_expires_after_n_bars() -> None:
    params = multi_params(pullback_require_thrust=False, pullback_wave_cooldown_bars=2)
    engine = machine(params)
    for item in trend_bars():
        engine.on_bar(item)
    ema_fast = engine._indicators.snapshot.ema_fast  # noqa: SLF001
    engine.on_bar(bar(120, ema_fast + 0.02, high=ema_fast + 0.06, low=ema_fast - 0.05))
    assert engine.position is not None
    engine.position = None  # имитируем стоп
    # Бар сразу после входа — блокирован; после истечения кулдауна вход снова возможен.
    blocked = engine.on_bar(bar(121, ema_fast + 0.02, high=ema_fast + 0.05, low=ema_fast - 0.05))
    assert not [event for event in blocked if event.kind == "entry"]
    allowed: list = []
    for index in range(4):
        allowed.extend(
            engine.on_bar(bar(122 + index, ema_fast + 0.02, high=ema_fast + 0.05, low=ema_fast - 0.05))
        )
        if [event for event in allowed if event.kind == "entry"]:
            break
    assert [event for event in allowed if event.kind == "entry"]


# ------------------------------------------------------------------ Engine C: комиссионный порог
def test_fade_defaults_are_stricter() -> None:
    params = BreakoutParams()
    assert params.fade_min_target_fee_ratio >= 5.0
    assert params.fade_rejection_close_ratio == pytest.approx(0.25)


def test_fade_requires_strong_rejection() -> None:
    engine = machine(multi_params(pullback_enabled=False, fade_min_target_fee_ratio=2.0))
    for item in chop_bars():
        engine.on_bar(item)
    center = engine._indicators.snapshot.ema_fast  # noqa: SLF001
    atr = engine._indicators.snapshot.atr_points  # noqa: SLF001
    upper = center + 2.2 * atr
    # Слабое закрытие в середине бара (позиция ≈0.5) — не отбой.
    weak = bar(120, upper - 0.3, high=upper + 0.4, low=upper - 0.75, open_=upper - 0.1)
    assert not [event for event in engine.on_bar(weak) if event.kind == "entry"]
    # Полосы для следующего бара считаются по предыдущему — пересчитываем их.
    center = engine._indicators.snapshot.ema_fast  # noqa: SLF001
    atr = engine._indicators.snapshot.atr_points  # noqa: SLF001
    upper = center + 2.2 * atr
    # Сильное закрытие в нижних 25 % — открывается.
    strong = bar(121, upper - 0.75, high=upper + 0.4, low=upper - 0.8, open_=upper - 0.2)
    assert [event for event in engine.on_bar(strong) if event.kind == "entry"]


def test_fade_fee_hurdle_blocks_small_targets() -> None:
    engine = machine(multi_params(pullback_enabled=False, fade_min_target_fee_ratio=200.0))
    for item in chop_bars():
        engine.on_bar(item)
    center = engine._indicators.snapshot.ema_fast  # noqa: SLF001
    atr = engine._indicators.snapshot.atr_points  # noqa: SLF001
    upper = center + 2.2 * atr
    events = engine.on_bar(bar(120, upper - 0.75, high=upper + 0.4, low=upper - 0.8, open_=upper - 0.2))
    assert not [event for event in events if event.kind == "entry"]
    assert engine.counters.fade_entries == 0


# ------------------------------------------------------------------ дедупликация активов
def test_underlying_family_maps_duplicates_together() -> None:
    assert underlying_family("CNYRUBF") == underlying_family("CNY_CONT") == "CNY"
    assert underlying_family("GAZPF") == underlying_family("GAZR_CONT") == "GAZP"
    assert underlying_family("USDRUBF") == underlying_family("Si_CONT") == "USD"
    assert underlying_family("GLDRUBF") == underlying_family("GOLDM_CONT") == "GOLD"
    assert underlying_family("IMOEXF") == underlying_family("MXI_CONT") == "INDEX"
    assert underlying_family("BRM_CONT") == "BRENT"


def test_deduplicate_keeps_best_per_family() -> None:
    metrics = {"CNYRUBF": 100.0, "CNY_CONT": 250.0, "GAZPF": 50.0, "GAZR_CONT": 10.0, "BRM_CONT": 70.0}
    kept = deduplicate(metrics)
    assert kept == {"BRENT": "BRM_CONT", "CNY": "CNY_CONT", "GAZP": "GAZPF"}


def test_deduplicate_is_deterministic_on_ties() -> None:
    metrics = {"CNYRUBF": 10.0, "CNY_CONT": 10.0}
    assert deduplicate(metrics, keep_order=["CNYRUBF", "CNY_CONT"]) == {"CNY": "CNYRUBF"}
    assert deduplicate(metrics, keep_order=["CNY_CONT", "CNYRUBF"]) == {"CNY": "CNY_CONT"}


# ------------------------------------------------------------------ портфель: сторож и тормоз
def trade_params(**overrides) -> BreakoutParams:
    """Параметры с отключёнными фильтрами, чтобы синтетика давала сделки."""
    return v32_params(volume_multiple=0.0, max_squeeze_ratio=0.0, min_atr_to_fee_ratio=0.0, **overrides)


def _portfolio(tickers, params, **settings_kwargs) -> SwingAccountBacktester:
    instruments = [
        (make_meta(t, price_step="0.01", step_amount="0.01", margin="1000"), params) for t in tickers
    ]
    settings_kwargs.setdefault("risk_pct_per_trade", 1.0)
    settings = SwingAccountSettings(capital_rub=50_000, **settings_kwargs)
    return SwingAccountBacktester(instruments, settings)


def _zigzag_candles(count: int = 400, amp: float = 0.6):
    from bot_a.backtest.candles import Candle
    from bot_a.utils.quotation import dec
    from tests.test_v32_swing import minute_ms

    candles = []
    price = 100.0
    for index in range(count):
        close = price + (amp if index % 2 else -amp)
        candles.append(
            Candle(
                time_ms=minute_ms(hour=10) + index * 30 * 60_000,
                open=dec(close - amp / 2),
                high=dec(close + amp / 4),
                low=dec(close - amp / 4),
                close=dec(close),
                volume=1000,
            )
        )
        price = close
    return candles


def test_health_guard_disabled_by_default() -> None:
    assert SwingAccountSettings().health_guard is False
    engine = _portfolio(["AAA"], trade_params())
    assert engine._settings.health_guard is False  # noqa: SLF001


def _guard_state(engine):
    health = {"AAA": []}
    paused = {"AAA": False}
    streak = {"AAA": 0}
    result = engine.run({"AAA": _trending_cycles()[:0]}).__class__()
    return health, paused, streak, result


def test_health_guard_pauses_after_consecutive_losses() -> None:
    engine = _portfolio(["AAA"], trade_params(), health_guard=True, consecutive_loss_limit=3)
    health, paused, streak, result = _guard_state(engine)
    settings = engine._settings  # noqa: SLF001
    for _ in range(3):
        engine._track_health("AAA", -100.0, settings, health, paused, streak, result)  # noqa: SLF001
    assert paused["AAA"] is True
    assert result.health_pauses == 1
    assert streak["AAA"] == 3


def test_health_guard_pauses_on_low_rolling_pf() -> None:
    engine = _portfolio(
        ["AAA"], trade_params(), health_guard=True, rolling_window_trades=4, rolling_pf_floor=0.7,
        consecutive_loss_limit=99,
    )
    health, paused, streak, result = _guard_state(engine)
    settings = engine._settings  # noqa: SLF001
    for value in (-100.0, 10.0, -100.0, 10.0):
        engine._track_health("AAA", value, settings, health, paused, streak, result)  # noqa: SLF001
    assert paused["AAA"] is True  # PF 20/200 = 0.1 < 0.70


def test_health_guard_does_not_pause_on_healthy_pf() -> None:
    engine = _portfolio(
        ["AAA"], trade_params(), health_guard=True, rolling_window_trades=4, rolling_pf_floor=0.7,
        consecutive_loss_limit=99,
    )
    health, paused, streak, result = _guard_state(engine)
    settings = engine._settings  # noqa: SLF001
    for value in (100.0, -50.0, 200.0, -50.0):
        engine._track_health("AAA", value, settings, health, paused, streak, result)  # noqa: SLF001
    assert paused["AAA"] is False
    assert result.health_pauses == 0


def test_health_guard_resumes_only_after_virtual_winner() -> None:
    engine = _portfolio(["AAA"], trade_params(), health_guard=True, consecutive_loss_limit=1)
    health, paused, streak, result = _guard_state(engine)
    settings = engine._settings  # noqa: SLF001
    engine._track_health("AAA", -100.0, settings, health, paused, streak, result)  # noqa: SLF001
    assert paused["AAA"] is True
    meta = make_meta("AAA", price_step="0.01", step_amount="0.01")
    params = trade_params()
    entry = _entry_event(100.0, BreakoutSide.LONG)
    # Виртуальный убыток — пауза сохраняется.
    engine._settle_virtual(  # noqa: SLF001
        "AAA", meta, params, entry, _virtual_position(100.0, BreakoutSide.LONG), _exit_event(99.0, BreakoutSide.LONG),
        settings, health, paused, streak, result,
    )
    assert paused["AAA"] is True and result.health_resumes == 0
    # Виртуальный плюс — инструмент возвращается в работу.
    entry_win = _entry_event(100.0, BreakoutSide.LONG)
    engine._settle_virtual(  # noqa: SLF001
        "AAA", meta, params, entry_win, _virtual_position(100.0, BreakoutSide.LONG),
        _exit_event(105.0, BreakoutSide.LONG), settings, health, paused, streak, result,
    )
    assert paused["AAA"] is False and result.health_resumes == 1


def test_drawdown_throttle_multiplier_curve() -> None:
    engine = _portfolio(["AAA"], trade_params(), drawdown_throttle=True, throttle_start_pct=10.0)
    assert engine._throttle_multiplier(50_000, 50_000) == pytest.approx(1.0)  # noqa: SLF001
    assert engine._throttle_multiplier(46_000, 50_000) == pytest.approx(1.0)  # noqa: SLF001 — 8 %
    assert engine._throttle_multiplier(40_000, 50_000) < 1.0  # noqa: SLF001 — 20 % просадки
    assert engine._throttle_multiplier(15_000, 50_000) == pytest.approx(0.4)  # noqa: SLF001 — пол
    off = _portfolio(["AAA"], trade_params(), drawdown_throttle=False)
    assert off._throttle_multiplier(10_000, 50_000) == pytest.approx(1.0)  # noqa: SLF001


def test_drawdown_throttle_scales_risk_down_without_stopping() -> None:
    params = trade_params(min_efficiency_ratio=0.0, close_in_range=1.0)
    engine = _portfolio(["AAA"], params, risk_pct_per_trade=2.0, drawdown_throttle=True,
                        throttle_start_pct=2.0)
    full = engine.size(engine._instruments[0][0], 1.0)  # noqa: SLF001
    engine._risk_multiplier = engine._throttle_multiplier(30_000, 50_000)  # noqa: SLF001
    reduced = engine.size(engine._instruments[0][0], 1.0)  # noqa: SLF001
    assert reduced <= full
    assert engine._risk_multiplier < 1.0  # noqa: SLF001


def _trending_cycles(cycles: int = 6, flat: int = 20, ramp: int = 4, level: float = 100.0):
    """Пила «боковик → импульс»: даёт пробойные входы на синтетике."""
    from bot_a.backtest.candles import Candle
    from bot_a.utils.quotation import dec
    from tests.test_v32_swing import minute_ms

    candles = []
    index = 0
    for _ in range(cycles):
        for _ in range(flat):
            candles.append(
                Candle(
                    time_ms=minute_ms(hour=10) + index * 30 * 60_000,
                    open=dec(level), high=dec(level + 0.1), low=dec(level - 0.1),
                    close=dec(level), volume=1000,
                )
            )
            index += 1
        for _ in range(ramp):
            level += 0.5
            candles.append(
                Candle(
                    time_ms=minute_ms(hour=10) + index * 30 * 60_000,
                    open=dec(level - 0.5), high=dec(level + 0.1), low=dec(level - 0.45),
                    close=dec(level), volume=1000,
                )
            )
            index += 1
    return candles


def _entry_event(price: float, side: BreakoutSide):
    from bot_a.engine.breakout_fsm import BreakoutEvent
    from bot_a.utils.quotation import dec

    return BreakoutEvent(kind="entry", time_ms=0, side=side, price=dec(price))


def _exit_event(price: float, side: BreakoutSide):
    from bot_a.engine.breakout_fsm import BreakoutEvent
    from bot_a.utils.quotation import dec

    return BreakoutEvent(kind="exit", time_ms=60_000, side=side, price=dec(price))


def _virtual_position(price: float, side: BreakoutSide):
    from bot_a.backtest.swing_portfolio import AccountPosition
    from bot_a.utils.quotation import dec

    return AccountPosition(
        ticker="AAA", side=side, quantity=1, margin_rub=0.0, entry_price=dec(price),
        entry_ms=0, entry_fee=dec("0"), atr_points=0.5, stop_points=1.0, virtual=True,
    )


# ------------------------------------------------------------------ фронт-контракт
UNIVERSE = {
    "front_by_month": {"BRM:2026-09": "BMV6", "BRM:2026-10": "BMX6", "GAZR:2026-10": "GZZ6"},
    "series": {
        "BMX6": {"asset": "BRM", "reason": "front 2026-10"},
        "GZZ6": {"asset": "GAZR", "reason": "front 2026-10"},
    },
}


def test_resolve_front_contract_uses_live_iss_list() -> None:
    current = {
        "BMZ6": {"secid": "BMZ6", "asset": "BRM", "last_trade": "2026-12-18", "margin": 1500.0},
        "BMX6": {"secid": "BMX6", "asset": "BRM", "last_trade": "2026-11-20", "margin": 1500.0},
        "BMV6": {"secid": "BMV6", "asset": "BRM", "last_trade": "2026-10-20", "margin": 1500.0},
    }
    assert resolve_front_contract("BRM_CONT", UNIVERSE, date(2026, 11, 6), current) == "BMX6"


def test_resolve_front_contract_prefers_perpetual() -> None:
    current = {
        "CNYRUBF": {"secid": "CNYRUBF", "asset": "CNY", "last_trade": "2100-01-01", "margin": 987.0},
        "CRZ6": {"secid": "CRZ6", "asset": "CNY", "last_trade": "2026-12-18", "margin": 1100.0},
    }
    assert resolve_front_contract("CNY_CONT", UNIVERSE, date(2026, 11, 6), current) == "CNYRUBF"


def test_resolve_front_contract_falls_back_to_cached_map() -> None:
    assert resolve_front_contract("BRM_CONT", UNIVERSE, date(2026, 11, 6)) == "BMX6"
    assert resolve_front_contract("GAZR_CONT", UNIVERSE, date(2026, 11, 6)) == "GZZ6"
    assert resolve_front_contract("UNKNOWN_CONT", UNIVERSE, date(2026, 11, 6)) is None


def test_resolve_front_contract_keeps_plain_tickers() -> None:
    assert resolve_front_contract("CNYRUBF", UNIVERSE) == "CNYRUBF"
    assert resolve_front_contract("GLDRUBF", UNIVERSE) == "GLDRUBF"


def test_resolve_swing_tickers_and_needs_resolution() -> None:
    tickers = ["CNYRUBF", "BRM_CONT", "GAZR_CONT"]
    mapping = resolve_swing_tickers(tickers, UNIVERSE, date(2026, 11, 6))
    assert mapping == {"CNYRUBF": "CNYRUBF", "BRM_CONT": "BMX6", "GAZR_CONT": "GZZ6"}
    assert needs_resolution(tickers) == ["BRM_CONT", "GAZR_CONT"]


# ------------------------------------------------------------------ интеграция
def test_backtester_keeps_continuous_names() -> None:
    """Бэктест работает с `*_CONT` как есть: резолв нужен только paper/live."""
    meta = make_meta("BRM_CONT", price_step="0.01", step_amount="0.8", margin="1600")
    result = BreakoutBacktester(meta, v32_params(), BreakoutSettings(capital_rub=50_000)).run(
        _zigzag_candles(200)
    )
    assert result.ticker == "BRM_CONT"


def test_multiregime_counters_exposed() -> None:
    engine = machine(multi_params())
    for item in trend_bars():
        engine.on_bar(item)
    payload = engine.counters.as_dict()
    assert "thrustSkips" in payload and "waveSkips" in payload
    assert isinstance(engine, MultiRegimeStateMachine)
