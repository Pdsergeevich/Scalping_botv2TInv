"""Тесты v4.5: сырая цена в комиссии FSM, макро-фильтр EMA_HTF, ресенси-гард, тиры риска.

Проверяем три причины провалов сентября/октября в v4.4:
гейты `ATR / Fee` и `Target / Fee` считались от back-adjusted цены, движки A/B
входили против макро-тренда, а отбор держал в корзине «остывшие» пары.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from bot_a.backtest.candles import Candle
from bot_a.backtest.walkforward import Signal, WalkForwardSettings, _window_stats, select_basket
from bot_a.engine.breakout_fsm import (
    Bar,
    BreakoutParams,
    BreakoutSide,
    IndicatorSnapshot,
    MomentumBreakoutStateMachine,
)
from bot_a.models.domain import InstrumentMetadata, MarketKind
from bot_a.utils.time_provider import MSK

DAY_MS = 86_400_000


def _ms(text: str, hour: int = 10) -> int:
    return int(datetime.fromisoformat(text).replace(hour=hour, tzinfo=MSK).timestamp() * 1000)


def _meta(ticker: str = "TEST_CONT", step: str = "1", step_price: str = "1", margin: int = 1000):
    return InstrumentMetadata(
        instrument_uid=ticker,
        ticker=ticker,
        class_code="SPBFUT",
        market=MarketKind.FUTURES,
        min_price_increment=Decimal(step),
        min_price_increment_amount=Decimal(step_price),
        lot_size=1,
        initial_margin_buy=Decimal(margin),
        initial_margin_sell=Decimal(margin),
        name=ticker,
    )


def _bar(price: str, adj: float = 0.0, high: str = "", low: str = "") -> Bar:
    return Bar(
        time_ms=_ms("2026-06-18"),
        open=Decimal(price),
        high=Decimal(high or price),
        low=Decimal(low or price),
        close=Decimal(price),
        volume=100,
        adj=adj,
    )


def _machine(**overrides) -> MomentumBreakoutStateMachine:
    return MomentumBreakoutStateMachine(_meta(), BreakoutParams().overrides(**overrides))


# ---------------------------------------------------------------- сырая цена


def test_raw_price_subtracts_adj() -> None:
    machine = _machine()
    bar = _bar("1100", adj=100.0)
    assert machine._raw_price(Decimal("1100"), bar) == Decimal("1000")


def test_raw_price_clamped_to_step() -> None:
    machine = _machine()
    bar = _bar("50", adj=100.0)  # adj больше цены — нельзя уйти в ноль/минус
    assert machine._raw_price(Decimal("50"), bar) == Decimal("1")


def test_raw_price_flag_off_reproduces_v44() -> None:
    machine = _machine(fee_on_raw_price=False)
    bar = _bar("1100", adj=100.0)
    assert machine._raw_price(Decimal("1100"), bar) == Decimal("1100")


def test_fee_gate_uses_raw_price_for_back_adjusted_series() -> None:
    """На сдвинутой серии комиссия берётся от сырой цены, а не от adjusted (x3 нотионал)."""
    fixed = _machine(fee_on_raw_price=True)
    broken = _machine(fee_on_raw_price=False)
    bar = _bar("3000", adj=2000.0)  # сырая цена 1000
    fee_fixed = fixed._fee_round_trip(fixed._raw_price(bar.close, bar), 1)
    fee_broken = broken._fee_round_trip(bar.close, 1)
    assert float(fee_fixed) < float(fee_broken) / 2.0


# ---------------------------------------------------------------- макро-фильтр


def _snapshot(close: float, ema_htf: float, slope: float, atr: float = 10.0, er: float = 0.25):
    return IndicatorSnapshot(
        ready=True,
        atr_points=atr,
        atr_rub=atr,
        ema_fast=close,
        ema_slow=close - 1.0,
        ema_htf=ema_htf,
        ema_htf_slope=slope,
        efficiency_ratio=er,
    )


def test_macro_filter_disabled_by_default() -> None:
    machine = _machine()
    bar = _bar("100", high="101", low="99")
    snapshot = _snapshot(100.0, ema_htf=200.0, slope=-50.0)
    assert machine._macro_ok(bar, snapshot, BreakoutSide.LONG) is True


def test_macro_filter_blocks_counter_trend_long() -> None:
    machine = _machine(htf_ema_bars=240)
    bar = _bar("100", high="101", low="99")
    # Цена ниже EMA_HTF и EMA_HTF падает — LONG против макро-тренда запрещён.
    assert machine._macro_ok(bar, _snapshot(100.0, 200.0, -50.0), BreakoutSide.LONG) is False
    assert machine.counters.macro_skips == 1


def test_macro_filter_allows_aligned_long() -> None:
    machine = _machine(htf_ema_bars=240)
    bar = _bar("300", high="301", low="299")
    assert machine._macro_ok(bar, _snapshot(300.0, 250.0, +12.0), BreakoutSide.LONG) is True


def test_macro_filter_slope_tolerance() -> None:
    machine = _machine(htf_ema_bars=240, htf_slope_tolerance_atr=0.05)
    bar = _bar("300", high="301", low="299")
    # EMA_HTF чуть ниже цены, наклон -0.4 пт при ATR 10 → допуск 0.5 пт, вход разрешён.
    assert machine._macro_ok(bar, _snapshot(300.0, 295.0, -0.4), BreakoutSide.LONG) is True
    # Тот же наклон, но цена под EMA_HTF — запрещено.
    assert machine._macro_ok(_bar("290", high="291", low="289"),
                             _snapshot(290.0, 295.0, -0.4), BreakoutSide.LONG) is False


def test_macro_filter_strong_impulse_overrides() -> None:
    machine = _machine(htf_ema_bars=240, htf_counter_er_override=0.40)
    bar = _bar("100", high="101", low="99")
    snapshot = _snapshot(100.0, 200.0, -50.0, er=0.55)
    assert machine._macro_ok(bar, snapshot, BreakoutSide.LONG) is True
    assert machine.counters.macro_skips == 0


def test_macro_filter_mirror_for_short() -> None:
    machine = _machine(htf_ema_bars=240)
    falling = _snapshot(100.0, 150.0, -20.0)
    rising = _snapshot(100.0, 50.0, +20.0)
    assert machine._macro_ok(_bar("100"), falling, BreakoutSide.SHORT) is True
    assert machine._macro_ok(_bar("100"), rising, BreakoutSide.SHORT) is False


# ---------------------------------------------------------------- ресенси-гард


def _signal(entry_day: str, exit_day: str, entry: str, exit_price: str, engine: str = "A") -> Signal:
    return Signal(
        ticker="AAA",
        engine=engine,
        entry_ms=_ms(entry_day),
        exit_ms=_ms(exit_day, 18),
        side=BreakoutSide.LONG,
        entry_price=Decimal(entry),
        exit_price=Decimal(exit_price),
        stop_points=1.0,
        conviction=1.0,
    )


def test_recency_guard_drops_cooled_pair() -> None:
    at_ms = _ms("2026-07-25")
    # Обе половины окна в плюсе (пара проходит базовый фильтр), но последние
    # 15 дней — минус: именно такой инструмент «остывает» к сентябрю.
    cooled = [
        _signal("2026-06-15", "2026-06-20", "100", "110"),
        _signal("2026-07-05", "2026-07-06", "100", "130"),
        _signal("2026-07-14", "2026-07-15", "100", "95"),
        _signal("2026-07-20", "2026-07-21", "100", "90"),
    ]
    healthy = [
        _signal("2026-06-15", "2026-06-20", "100", "110"),
        _signal("2026-06-25", "2026-06-30", "100", "110"),
        _signal("2026-07-14", "2026-07-15", "100", "110"),
        _signal("2026-07-20", "2026-07-21", "100", "110"),
    ]
    metas = {"AAA": _meta("AAA")}
    without_guard = select_basket({("AAA", "A"): cooled}, metas, at_ms, lookback_days=45,
                                  top=4, min_pf=1.0, min_trades=2, deduplicate=False)
    assert [item["ticker"] for item in without_guard] == ["AAA"]

    with_guard = select_basket({("AAA", "A"): cooled}, metas, at_ms, lookback_days=45,
                               top=4, min_pf=1.0, min_trades=2, deduplicate=False,
                               recent_days=15)
    assert with_guard == []

    kept = select_basket({("AAA", "A"): healthy}, metas, at_ms, lookback_days=45,
                         top=4, min_pf=1.0, min_trades=2, deduplicate=False,
                         recent_days=15)
    assert [item["ticker"] for item in kept] == ["AAA"]


def test_recency_guard_reports_recent_net() -> None:
    at_ms = _ms("2026-07-25")
    series = [_signal("2026-06-15", "2026-06-20", "100", "110"),
              _signal("2026-07-20", "2026-07-21", "100", "108")]
    kept = select_basket({("AAA", "A"): series}, {"AAA": _meta("AAA")}, at_ms,
                         lookback_days=45, top=4, min_pf=1.0, min_trades=2,
                         deduplicate=False, recent_days=15)
    assert kept and kept[0]["recent_net"] > 0


# ---------------------------------------------------------------- тиры риска


def test_v45_defaults() -> None:
    settings = WalkForwardSettings()
    assert settings.recent_days == 15
    assert settings.conviction_strong_boost > 1.0
    assert settings.conviction_border_cut < 1.0
    assert 0.0 < settings.conviction_border_er < settings.conviction_strong_er


def test_signal_records_er() -> None:
    signal = _signal("2026-06-15", "2026-06-20", "100", "110")
    assert signal.er == 0.0  # по умолчанию не заполнен
    signal.er = 0.41
    assert signal.er == pytest.approx(0.41)


def test_window_stats_uses_raw_price_for_fees() -> None:
    """Комиссия считается от `price − adj`, поэтому на сдвинутой серии она та же."""
    meta = _meta("AAA")
    plain = [_signal("2026-06-15", "2026-06-20", "1000", "1010")]
    shifted = [
        Signal(
            ticker="AAA",
            engine="A",
            entry_ms=_ms("2026-06-15"),
            exit_ms=_ms("2026-06-20", 18),
            side=BreakoutSide.LONG,
            entry_price=Decimal("3000"),
            exit_price=Decimal("3010"),
            stop_points=1.0,
            conviction=1.0,
            adj=2000.0,
        )
    ]
    window = (_ms("2026-06-01"), _ms("2026-07-01"), meta)
    base = _window_stats(plain, window[0], window[1], window[2])
    moved = _window_stats(shifted, window[0], window[1], window[2])
    assert moved["net"] == pytest.approx(base["net"], abs=1e-6)


def test_candle_adj_reaches_bar() -> None:
    candle = Candle(_ms("2026-06-18"), Decimal("100"), Decimal("101"), Decimal("99"),
                    Decimal("100"), 5, adj=3.5)
    bar = Bar(time_ms=candle.time_ms, open=candle.open, high=candle.high, low=candle.low,
              close=candle.close, volume=int(candle.volume),
              adj=float(getattr(candle, "adj", 0.0) or 0.0))
    assert bar.adj == pytest.approx(3.5)
