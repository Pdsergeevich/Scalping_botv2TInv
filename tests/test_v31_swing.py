"""Тесты v3.1: перенос позиции через ночь и вход по касанию уровня.

Основание — §8 отчёта `forts_momentum_breakout_v3_0.md`:

* принудительное закрытие `18:35 MSK` срезало тренды `CNYRUBF`/`GLDRUBF` посреди
  американской сессии; на FORTS перенос через ночь не стоит маржинального процента;
* вход по `close` бара пробоя «догонял» движение (до `0.35 ATR`), а `maxEntryExtensionAtr`
  отбрасывал самые быстрые пробои; вход стоп-заявкой на `H + 1 шаг` устраняет оба эффекта;
* комиссия Т-Инвестиций на фьючерсы: `Premium` — 0.025 % до 12 млн ₽ оборота в день,
  0.02 % до 17 млн, 0.015 % свыше; «дополнительный список» — 0.06 %/0.08 % независимо от оборота.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from bot_a.backtest.breakout_backtest import (
    BreakoutBacktester,
    BreakoutResult,
    BreakoutSettings,
    BreakoutTrade,
    aggregate,
    portfolio_curve,
)
from bot_a.economics.fees import Tariff, TinkoffFeeCalculator
from bot_a.engine.breakout_fsm import (
    Bar,
    BreakoutExitReason,
    BreakoutParams,
    BreakoutSide,
    MomentumBreakoutStateMachine,
)
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import MSK, msk_to_ms
from tests.helpers import make_meta


def minute_ms(day: str = "2026-05-04", hour: int = 11, minute: int = 0) -> int:
    date = datetime.fromisoformat(day).date()
    return msk_to_ms(datetime(date.year, date.month, date.day, hour, minute, tzinfo=MSK))


def bar(
    index: int,
    close: float,
    *,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    volume: int = 1000,
    start_ms: int | None = None,
) -> Bar:
    return Bar(
        time_ms=(start_ms if start_ms is not None else minute_ms()) + index * 60_000,
        open=dec(open_ if open_ is not None else close),
        high=dec(high if high is not None else close),
        low=dec(low if low is not None else close),
        close=dec(close),
        volume=int(volume),
    )


def swing_params(**overrides) -> BreakoutParams:
    base = dict(
        donchian_bars=10,
        atr_bars=5,
        squeeze_median_bars=5,
        max_squeeze_ratio=0.0,
        max_relative_squeeze=0.0,
        ema_fast_bars=2,
        ema_slow_bars=4,
        volume_lookback=5,
        volume_multiple=0.5,
        close_in_range=0.6,
        min_atr_to_fee_ratio=0.0,
        obi_long_threshold=0.2,
        obi_short_threshold=-0.2,
        breakout_buffer_steps=1,
        session_windows="00:00-23:59",
        trade_until="18:35",
        initial_stop_atr_multiple=1.2,
        min_stop_steps=1,
        failed_breakout_minutes=0,
        failed_breakout_bars=2,
        trail_trigger_atr=1.8,
        trail_atr_multiple=2.0,
        take_profit_atr=0.0,
        max_consecutive_stops=2,
        cooldown_minutes=45,
        slippage_steps=1,
    )
    base.update(overrides)
    return BreakoutParams(**base)


def machine(params: BreakoutParams, ticker: str = "TEST") -> MomentumBreakoutStateMachine:
    return MomentumBreakoutStateMachine(
        make_meta(ticker, price_step="0.01", step_amount="0.01"), params
    )


def long_bars(count: int = 24, start_ms: int | None = None, price: float = 100.0) -> list[Bar]:
    """Спокойный канал, затем бар пробоя (объём, закрытие у максимума)."""
    from tests.test_v30_breakout import flat_bars

    bars = flat_bars(count, price=price, amplitude=0.2, volume=1000, start_ms=start_ms)
    close = price + 0.6
    bars.append(
        bar(
            count,
            close,
            high=close + 0.05,
            low=price + 0.15,
            open_=price + 0.2,
            volume=5000,
            start_ms=start_ms,
        )
    )
    return bars


def rising_bars(count: int = 24, start_ms: int | None = None, price: float = 100.0) -> list[Bar]:
    """Плавно растущий ряд: EMA(fast) выше EMA(slow), закрытия в верхней части диапазона."""
    bars: list[Bar] = []
    for index in range(count):
        close = price + index * 0.05
        bars.append(
            bar(
                index,
                close,
                high=close + 0.05,
                low=close - 0.15,
                open_=close - 0.02,
                volume=1000,
                start_ms=start_ms,
            )
        )
    return bars


# --------------------------------------------------------------- перенос через ночь
def test_intraday_mode_closes_position_at_trade_until() -> None:
    engine = machine(swing_params(allow_overnight_hold=False))
    entry_price = 0.0
    for item in long_bars():
        engine.on_bar(item)
    assert engine.position is not None
    entry_price = float(engine.position.entry_price)
    events = engine.on_bar(
        bar(0, entry_price + 0.3, high=entry_price + 0.4, low=entry_price + 0.2,
            start_ms=minute_ms(hour=18, minute=40))
    )
    reasons = [event.reason for event in events if event.kind == "exit"]
    assert BreakoutExitReason.SESSION_CLOSE in reasons
    assert engine.position is None


def test_overnight_mode_keeps_position_past_trade_until() -> None:
    engine = machine(swing_params(allow_overnight_hold=True))
    for item in long_bars():
        engine.on_bar(item)
    assert engine.position is not None
    entry_price = float(engine.position.entry_price)
    # 18:40 — время, когда интрадей-режим закрыл бы сделку; здесь позиция остаётся.
    events = engine.on_bar(
        bar(0, entry_price + 0.5, high=entry_price + 0.6, low=entry_price + 0.3,
            start_ms=minute_ms(hour=18, minute=40))
    )
    assert not [event for event in events if event.kind == "exit"]
    assert engine.position is not None
    # Следующий день: позиция жива и ведётся трейлом.
    events = engine.on_bar(
        bar(0, entry_price + 1.0, high=entry_price + 1.2, low=entry_price + 0.8,
            start_ms=minute_ms(day="2026-05-05", hour=10, minute=30))
    )
    assert engine.position is not None


def test_overnight_loser_is_cut_before_clearing() -> None:
    params = swing_params(
        allow_overnight_hold=True, close_losers_before_clearing=True, loser_close_bars=1
    )
    engine = machine(params)
    for item in long_bars():
        engine.on_bar(item)
    assert engine.position is not None
    entry_price = float(engine.position.entry_price)
    # Три бара в минусе, затем 18:40 — срез убыточной позиции перед клирингом.
    for index in range(3):
        engine.on_bar(
            bar(index, entry_price - 0.15, high=entry_price - 0.05, low=entry_price - 0.25)
        )
    assert engine.position is not None
    events = engine.on_bar(
        bar(0, entry_price - 0.2, high=entry_price - 0.1, low=entry_price - 0.3,
            start_ms=minute_ms(hour=18, minute=40))
    )
    reasons = [event.reason for event in events if event.kind == "exit"]
    assert BreakoutExitReason.CLEARING in reasons


def test_overnight_winner_is_not_cut_before_clearing() -> None:
    params = swing_params(
        allow_overnight_hold=True, close_losers_before_clearing=True, loser_close_bars=1
    )
    engine = machine(params)
    for item in long_bars():
        engine.on_bar(item)
    entry_price = float(engine.position.entry_price)
    events = engine.on_bar(
        bar(0, entry_price + 0.5, high=entry_price + 0.6, low=entry_price + 0.4,
            start_ms=minute_ms(hour=18, minute=40))
    )
    assert not [event for event in events if event.kind == "exit"]
    assert engine.position is not None


def test_max_hold_hours_forces_exit() -> None:
    engine = machine(swing_params(allow_overnight_hold=True, max_hold_hours=2.0))
    for item in long_bars():
        engine.on_bar(item)
    entry_price = float(engine.position.entry_price)
    events = engine.on_bar(
        bar(0, entry_price + 0.1, high=entry_price + 0.2, low=entry_price + 0.05,
            start_ms=minute_ms(hour=14, minute=0))
    )
    reasons = [event.reason for event in events if event.kind == "exit"]
    assert BreakoutExitReason.MAX_HOLD in reasons


def test_gap_through_stop_fills_at_open() -> None:
    engine = machine(swing_params(allow_overnight_hold=True))
    for item in long_bars():
        engine.on_bar(item)
    position = engine.position
    assert position is not None
    stop = float(position.stop_price)
    events = engine.on_bar(
        bar(0, stop - 2.0, high=stop - 1.5, low=stop - 3.0, open_=stop - 1.8,
            start_ms=minute_ms(day="2026-05-05", hour=10, minute=0))
    )
    exits = [event for event in events if event.kind == "exit"]
    assert exits
    assert float(exits[0].price) == pytest.approx(stop - 1.8)


# --------------------------------------------------------------- вход по касанию уровня
def consolidated_bars(start_ms: int | None = None) -> list[Bar]:
    """«Бычий флаг»: рост (тренд вверх) → узкая консолидация ниже пика.

    Высокие максимумы флага не доходят до `пик + 1 шаг`, поэтому уровень взведён,
    но касания нет — идеальная заготовка для проверки входа по стоп-уровню.
    """
    bars: list[Bar] = []
    index = 0
    for step in range(16):
        close = 100.0 + step * 0.1
        bars.append(
            bar(
                index,
                close,
                high=close + 0.1,
                low=close - 0.05,
                open_=close - 0.02,
                volume=1000,
                start_ms=start_ms,
            )
        )
        index += 1
    for step in range(6):
        close = 101.45 + step * 0.005
        bars.append(
            bar(
                index,
                close,
                high=close + 0.02,
                low=close - 0.12,
                open_=close - 0.04,
                volume=1000,
                start_ms=start_ms,
            )
        )
        index += 1
    return bars


def _armed_engine(**overrides) -> MomentumBreakoutStateMachine:
    params = swing_params(entry_mode="stop_level", **overrides)
    return machine(params)


def test_stop_level_entry_triggers_on_touch_of_previous_high() -> None:
    engine = _armed_engine()
    for item in consolidated_bars():
        engine.on_bar(item)
    assert engine.position is None
    assert engine._armed_side is BreakoutSide.LONG  # noqa: SLF001
    trigger = float(engine._armed_trigger)  # noqa: SLF001
    assert trigger == pytest.approx(101.61)  # пик 101.60 + 1 шаг (0.01)
    # Бар касается уровня внутри диапазона: вход по триггеру, а не по закрытию.
    touching = bar(
        28,
        trigger + 0.05,
        high=trigger + 0.2,
        low=trigger - 0.05,
        open_=trigger - 0.05,
        volume=900,
    )
    events = engine.on_bar(touching)
    entries = [event for event in events if event.kind == "entry"]
    assert entries, "ожидался вход по касанию уровня"
    assert float(entries[0].price) == pytest.approx(trigger)
    assert engine.position is not None


def test_stop_level_entry_skips_gap_above_trigger() -> None:
    engine = _armed_engine(max_entry_extension_atr=0.35)
    for item in consolidated_bars():
        engine.on_bar(item)
    trigger = float(engine._armed_trigger)  # noqa: SLF001
    atr = engine._armed_atr  # noqa: SLF001
    gapped = bar(
        28,
        trigger + 5 * atr,
        high=trigger + 6 * atr,
        low=trigger + 4 * atr,
        open_=trigger + 5 * atr,
    )
    events = engine.on_bar(gapped)
    assert not [event for event in events if event.kind == "entry"]
    assert engine.counters.extension_skips >= 1
    assert engine.position is None


def test_stop_level_entry_not_triggered_without_touch() -> None:
    engine = _armed_engine()
    for item in consolidated_bars():
        engine.on_bar(item)
    trigger = float(engine._armed_trigger)  # noqa: SLF001
    events = engine.on_bar(
        bar(28, trigger - 0.3, high=trigger - 0.1, low=trigger - 0.5, open_=trigger - 0.2)
    )
    assert not [event for event in events if event.kind == "entry"]
    assert engine.position is None


def bear_flag_bars(start_ms: int | None = None) -> list[Bar]:
    """«Медвежий флаг»: падение → узкая консолидация выше минимума (для шорт-уровня)."""
    bars: list[Bar] = []
    index = 0
    for step in range(16):
        close = 100.0 - step * 0.1
        bars.append(
            bar(
                index,
                close,
                high=close + 0.05,
                low=close - 0.1,
                open_=close + 0.02,
                volume=1000,
                start_ms=start_ms,
            )
        )
        index += 1
    for step in range(6):
        close = 98.55 - step * 0.005
        bars.append(
            bar(
                index,
                close,
                high=close + 0.12,
                low=close - 0.02,
                open_=close + 0.04,
                volume=1000,
                start_ms=start_ms,
            )
        )
        index += 1
    return bars


def test_stop_level_short_side_arms_and_triggers() -> None:
    engine = _armed_engine()
    for item in bear_flag_bars():
        engine.on_bar(item)
    assert engine.position is None
    assert engine._armed_side is BreakoutSide.SHORT  # noqa: SLF001
    trigger = float(engine._armed_trigger)  # noqa: SLF001
    assert trigger == pytest.approx(98.39)  # минимум 98.40 - 1 шаг
    events = engine.on_bar(
        bar(22, trigger - 0.03, high=trigger + 0.04, low=trigger - 0.1, open_=trigger + 0.02)
    )
    entries = [event for event in events if event.kind == "entry"]
    assert entries
    assert entries[0].side is BreakoutSide.SHORT


def test_stop_level_mode_through_backtester_matches_direction() -> None:
    from bot_a.backtest.candles import Candle

    meta = make_meta("SR", price_step="0.01", step_amount="0.01", margin="1000")
    bars = consolidated_bars()
    last_close = float(bars[-1].close)
    trigger = float(bars[15].high) + 0.01  # пик флага + 1 шаг
    # Пробой: открытие чуть ниже триггера (без гэпа), затем тренд вверх.
    bars.append(
        bar(22, trigger + 0.05, high=trigger + 0.12, low=trigger - 0.03, open_=trigger - 0.02,
            volume=1200)
    )
    price = trigger + 0.05
    for index in range(1, 10):
        close = price + 0.3
        bars.append(bar(22 + index, close, high=close + 0.05, low=price - 0.05, open_=price,
                        volume=1200))
        price = close
    candles = [
        Candle(
            time_ms=item.time_ms,
            open=item.open,
            high=item.high,
            low=item.low,
            close=item.close,
            volume=item.volume,
        )
        for item in bars
    ]
    backtester = BreakoutBacktester(
        meta,
        swing_params(entry_mode="stop_level"),
        BreakoutSettings(capital_rub=50_000, risk_pct_per_trade=1.0),
    )
    result = backtester.run(candles)
    assert result.trades, "вход по уровню должен дать сделку"
    assert result.trades[0].side is BreakoutSide.LONG


# --------------------------------------------------------------- комиссия (проверено по Т-Банку)
def test_premium_futures_tiers_match_broker_help() -> None:
    calc = TinkoffFeeCalculator(Tariff.PREMIUM)
    assert calc.current_rate == Decimal("0.00025")  # до 12 млн ₽ оборота по фьючерсам
    calc.register_notional(Decimal("12000001"))
    assert calc.current_rate == Decimal("0.0002")  # 12–17 млн
    calc.register_notional(Decimal("5000000"))
    assert calc.current_rate == Decimal("0.00015")  # свыше 17 млн


def test_additional_list_futures_flat_rate() -> None:
    calc = TinkoffFeeCalculator(Tariff.PREMIUM, additional_list_tickers=["BR"])
    assert calc.fee_for_notional(Decimal("100000"), "BR") == Decimal("60.00")
    trader = TinkoffFeeCalculator(Tariff.TRADER, additional_list_tickers=["BR"])
    assert trader.fee_for_notional(Decimal("100000"), "BR") == Decimal("80.00")


# --------------------------------------------------------------- метрики портфеля
def _trade(net: float, day_ms: int, close_ms: int) -> BreakoutTrade:
    return BreakoutTrade(
        ticker="X",
        side=BreakoutSide.LONG,
        quantity=1,
        entry_ms=day_ms,
        exit_ms=close_ms,
        entry_price=dec(100),
        exit_price=dec(100),
        gross_rub=net,
        fees_rub=0.0,
        net_rub=net,
        reason=BreakoutExitReason.TRAIL_STOP,
        atr_points=1.0,
        mfe_points=1.0,
        hold_minutes=60.0,
    )


def test_portfolio_curve_and_sharpe() -> None:
    base = minute_ms(day="2026-05-04", hour=12)
    trades = [
        _trade(100.0, base, base + 3_600_000),
        _trade(-40.0, base + 86_400_000, base + 90_000_000),
        _trade(60.0, base + 172_800_000, base + 176_400_000),
        _trade(30.0, base + 259_200_000, base + 262_800_000),
    ]
    result = BreakoutResult(ticker="X", trades=trades)
    curve = portfolio_curve([result], 50_000.0)
    assert curve["netRub"] == pytest.approx(150.0)
    assert curve["maxDrawdownRub"] == pytest.approx(40.0)
    assert curve["maxDrawdownPct"] == pytest.approx(0.08)
    assert curve["days"] == 4
    assert curve["sharpe"] is not None and curve["sharpe"] > 0
    assert list(curve["monthly"]) == ["2026-05"]
    assert curve["monthly"]["2026-05"] == pytest.approx(150.0)
    summary = aggregate([result], 50_000.0)
    assert summary["netRub"] == pytest.approx(150.0)
    assert summary["tickers"] == ["X"]


def test_empty_portfolio_curve_is_safe() -> None:
    curve = portfolio_curve([BreakoutResult(ticker="X")], 50_000.0)
    assert curve["netRub"] == 0.0
    assert curve["sharpe"] is None
    assert curve["points"] == []
