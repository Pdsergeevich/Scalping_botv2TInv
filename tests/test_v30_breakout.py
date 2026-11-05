"""Тесты v3.0: пробой сжатия волатильности на фьючерсах FORTS.

Проверяем три уровня:

1. **Индикаторы** (`BreakoutIndicators`): Donchian по предыдущим барам, ATR Уайлдера,
   EMA, объём предыдущих баров и коэффициент сжатия `W / (ATR × √N)`;
2. **ФСМ** (`MomentumBreakoutStateMachine`): вход только при пробое с объёмом, трендом,
   закрытием в верхних/нижних 30 % и OBI; стоп 1.2 ATR, breakeven-ratchet, Chandelier,
   срез неудачного пробоя, кулдаун после стопов, закрытие сессии;
3. **Экономика** (`BreakoutBacktester`): сайзинг от риска и ГО, проскальзывание в 1 шаг,
   комиссия Premium за обе стороны, метрики (PF, Payoff, Max DD).
"""

from __future__ import annotations

import math
from datetime import datetime
from decimal import Decimal

import pytest

from bot_a.backtest.breakout_backtest import (
    BreakoutBacktester,
    BreakoutResult,
    BreakoutSettings,
    BreakoutTrade,
)
from bot_a.config import AppConfig
from bot_a.economics.fees import Tariff
from bot_a.engine.breakout_fsm import (
    Bar,
    BreakoutExitReason,
    BreakoutIndicators,
    BreakoutParams,
    BreakoutSide,
    MomentumBreakoutStateMachine,
    in_windows,
    parse_windows,
)
from bot_a.services.breakout_worker import BreakoutBarBuilder, BreakoutLiveWorker
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import MSK, msk_to_ms
from tests.helpers import make_book, make_meta

# --------------------------------------------------------------------- утилиты


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
    price = dec(close)
    return Bar(
        time_ms=(start_ms if start_ms is not None else minute_ms()) + index * 60_000,
        open=dec(open_ if open_ is not None else close),
        high=dec(high if high is not None else close),
        low=dec(low if low is not None else close),
        close=price,
        volume=int(volume),
    )


def flat_bars(
    count: int,
    price: float = 100.0,
    *,
    amplitude: float = 0.2,
    volume: int = 1000,
    start_ms: int | None = None,
) -> list[Bar]:
    """Пилообразные бары без тренда — канал сужается, пробоя нет."""
    result: list[Bar] = []
    for index in range(count):
        wobble = amplitude if index % 2 == 0 else -amplitude
        close = price + wobble
        result.append(
            bar(
                index,
                close,
                high=close + amplitude / 2,
                low=close - amplitude / 2,
                open_=close - wobble,
                volume=volume,
                start_ms=start_ms,
            )
        )
    return result


def fast_params(**overrides) -> BreakoutParams:
    """Параметры, при которых индикаторы готовы быстро и фильтр сжатия выключен."""
    base = dict(
        donchian_bars=10,
        atr_bars=5,
        squeeze_median_bars=5,
        max_squeeze_ratio=0.0,
        ema_fast_bars=3,
        ema_slow_bars=5,
        volume_lookback=5,
        volume_multiple=1.5,
        close_in_range=0.3,
        min_atr_to_fee_ratio=0.0,
        obi_long_threshold=0.2,
        obi_short_threshold=-0.2,
        breakout_buffer_steps=1,
        session_windows="00:00-23:59",
        trade_until="23:59",
        initial_stop_atr_multiple=1.2,
        min_stop_steps=1,
        failed_breakout_minutes=12,
        trail_trigger_atr=1.8,
        trail_atr_multiple=2.0,
        take_profit_atr=0.0,
        max_consecutive_stops=2,
        cooldown_minutes=45,
        slippage_steps=1,
    )
    base.update(overrides)
    return BreakoutParams(**base)


def machine(params: BreakoutParams, meta=None) -> MomentumBreakoutStateMachine:
    return MomentumBreakoutStateMachine(meta or make_meta("TEST", price_step="0.01", step_amount="0.01"), params)


# --------------------------------------------------------------------- окна


def test_parse_windows_and_membership() -> None:
    windows = parse_windows("10:15-13:45,14:10-18:20")
    assert windows == ((615, 825), (850, 1100))
    assert in_windows(minute_ms(hour=10, minute=14), windows) is False
    assert in_windows(minute_ms(hour=10, minute=15), windows) is True
    assert in_windows(minute_ms(hour=13, minute=44), windows) is True
    assert in_windows(minute_ms(hour=13, minute=45), windows) is False
    assert in_windows(minute_ms(hour=14, minute=10), windows) is True
    assert in_windows(minute_ms(hour=18, minute=19), windows) is True
    assert in_windows(minute_ms(hour=18, minute=20), windows) is False
    assert in_windows(minute_ms(hour=18, minute=21), windows) is False


# --------------------------------------------------------------------- индикаторы


def test_donchian_uses_previous_bars_only() -> None:
    params = fast_params(donchian_bars=5, max_squeeze_ratio=0.0)
    indicators = BreakoutIndicators(params, Decimal(1))
    bars = [bar(index, 100.0 + index) for index in range(12)]
    for item in bars:
        snapshot = indicators.update(item)
    # Канал — по пяти ПРЕДЫДУЩИМ барам (6..10), текущий (11-й) не учтён.
    assert snapshot.donchian_high == pytest.approx(110.0)
    assert snapshot.donchian_low == pytest.approx(106.0)
    assert snapshot.channel_width == pytest.approx(4.0)
    assert snapshot.ready is True


def test_atr_wilder_and_ema_and_volume_lag() -> None:
    params = fast_params(donchian_bars=3, atr_bars=3, volume_lookback=3)
    indicators = BreakoutIndicators(params, Decimal(10))
    snapshot = None
    volumes = [100, 100, 400]
    for index, volume in enumerate(volumes):
        snapshot = indicators.update(
            bar(index, 100.0, high=101.0, low=99.0, volume=volume)
        )
    # ATR: TR = 2, сглаживание 1/3 остаётся 2 при постоянном диапазоне.
    assert snapshot.atr_points == pytest.approx(2.0, abs=1e-6)
    assert snapshot.atr_rub == pytest.approx(20.0, abs=1e-6)
    # Средний объём считается по ПРЕДЫДУЩИМ барам: (100 + 100) / 2.
    assert snapshot.volume_avg == pytest.approx(100.0)
    assert snapshot.ema_fast == pytest.approx(100.0)


def test_squeeze_ratio_formula() -> None:
    params = fast_params(donchian_bars=4, atr_bars=2, max_squeeze_ratio=1.6)
    indicators = BreakoutIndicators(params, Decimal(1))
    for index in range(6):
        snapshot = indicators.update(bar(index, 100.0, high=100.5, low=99.5))
    expected = snapshot.channel_width / (snapshot.atr_points * math.sqrt(4))
    assert snapshot.squeeze_ratio == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------- вход


def _long_breakout_bars(count: int = 24) -> list[Bar]:
    """Спокойный канал, затем бар пробоя с объёмом и закрытием у максимума."""
    bars = flat_bars(count, price=100.0, amplitude=0.2, volume=1000)
    break_close = 100.6
    bars.append(
        bar(
            count,
            break_close,
            high=break_close + 0.05,
            low=100.15,
            open_=100.2,
            volume=5000,
        )
    )
    return bars


def test_long_entry_requires_volume_trend_and_close_position() -> None:
    engine = machine(fast_params())
    events = []
    for item in _long_breakout_bars():
        events.extend(engine.on_bar(item, obi=0.5))
    entries = [event for event in events if event.kind == "entry"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.side is BreakoutSide.LONG
    assert engine.position is not None
    stop_distance = float(engine.position.entry_price - engine.position.stop_price)
    assert stop_distance >= 1.2 * engine.position.atr_points - 1e-9


def test_entry_blocked_without_volume_thrust() -> None:
    bars = _long_breakout_bars()
    bars[-1].volume = 100  # нет объёмного всплеска
    engine = machine(fast_params())
    events = []
    for item in bars:
        events.extend(engine.on_bar(item, obi=0.5))
    assert not [event for event in events if event.kind == "entry"]
    assert engine.counters.volume_skips >= 1


def test_entry_blocked_by_wick_and_obi() -> None:
    bars = _long_breakout_bars()
    bars[-1].high = dec(101.2)  # закрытие в нижней части диапазона — ложный пробой
    engine = machine(fast_params())
    events = []
    for item in bars:
        events.extend(engine.on_bar(item, obi=0.5))
    assert not [event for event in events if event.kind == "entry"]
    assert engine.counters.wick_skips >= 1

    bars = _long_breakout_bars()
    engine = machine(fast_params())
    events = []
    for item in bars:
        events.extend(engine.on_bar(item, obi=-0.6))  # стакан против сделки
    assert not [event for event in events if event.kind == "entry"]
    assert engine.counters.obi_skips >= 1


def test_entry_blocked_by_squeeze_filter() -> None:
    engine = machine(fast_params(max_squeeze_ratio=0.01))  # невозможное сжатие
    events = []
    for item in _long_breakout_bars():
        events.extend(engine.on_bar(item, obi=0.5))
    assert not [event for event in events if event.kind == "entry"]
    assert engine.counters.squeeze_skips >= 1


def test_entry_blocked_by_relative_squeeze() -> None:
    engine = machine(fast_params(max_squeeze_ratio=0.0, max_relative_squeeze=0.01))
    events = []
    for item in _long_breakout_bars():
        events.extend(engine.on_bar(item, obi=0.5))
    assert not [event for event in events if event.kind == "entry"]
    assert engine.counters.squeeze_skips >= 1


def test_entry_blocked_outside_session_window() -> None:
    params = fast_params(session_windows="10:15-13:45")
    engine = machine(params)
    bars = _long_breakout_bars()
    events = []
    for index, item in enumerate(bars):
        item.time_ms = minute_ms(hour=9, minute=0) + index * 60_000
        events.extend(engine.on_bar(item, obi=0.5))
    assert not [event for event in events if event.kind == "entry"]
    assert engine.counters.window_skips >= 1


def test_entry_blocked_by_fee_hurdle() -> None:
    engine = machine(fast_params(min_atr_to_fee_ratio=1_000.0))
    events = []
    for item in _long_breakout_bars():
        events.extend(engine.on_bar(item, obi=0.5))
    assert not [event for event in events if event.kind == "entry"]
    assert engine.counters.fee_skips >= 1


def test_short_entry_on_downside_breakout() -> None:
    bars = flat_bars(24, price=100.0, amplitude=0.2)
    close = 99.4
    bars.append(bar(24, close, high=99.85, low=close - 0.05, open_=99.8, volume=5000))
    engine = machine(fast_params())
    events = []
    for item in bars:
        events.extend(engine.on_bar(item, obi=-0.5))
    entries = [event for event in events if event.kind == "entry"]
    assert len(entries) == 1
    assert entries[0].side is BreakoutSide.SHORT


# --------------------------------------------------------------------- выходы


def _open_long(engine: MomentumBreakoutStateMachine) -> float:
    events: list = []
    for item in _long_breakout_bars():
        events.extend(engine.on_bar(item, obi=0.5))
    assert engine.position is not None
    return float(engine.position.entry_price)


def test_hard_stop_is_hit() -> None:
    engine = machine(fast_params())
    entry_price = _open_long(engine)
    events = engine.on_bar(
        bar(20, entry_price - 5.0, high=entry_price - 1.0, low=entry_price - 6.0),
        obi=0.0,
    )
    exits = [event for event in events if event.kind == "exit"]
    assert len(exits) == 1
    assert exits[0].reason is BreakoutExitReason.HARD_STOP
    assert engine.position is None


def test_failed_breakout_time_cut() -> None:
    engine = machine(fast_params(failed_breakout_minutes=3))
    entry_price = _open_long(engine)
    exits: list = []
    for index in range(1, 5):
        exits.extend(
            engine.on_bar(
                bar(24 + index, entry_price - 0.1, high=entry_price + 0.05, low=entry_price - 0.2),
                obi=0.0,
            )
        )
    reasons = [event.reason for event in exits if event.kind == "exit"]
    assert BreakoutExitReason.FAILED_BREAKOUT in reasons


def test_breakeven_ratchet_moves_stop_above_fees() -> None:
    engine = machine(fast_params())
    entry_price = _open_long(engine)
    atr = engine.position.atr_points
    # Первый бар фиксирует MFE 1.35 ATR, второй — включает breakeven-ratchet.
    engine.on_bar(
        bar(24, entry_price + 1.3 * atr, high=entry_price + 1.35 * atr, low=entry_price + 0.9 * atr),
        obi=0.0,
    )
    engine.on_bar(
        bar(25, entry_price + 1.3 * atr, high=entry_price + 1.3 * atr, low=entry_price + 1.1 * atr),
        obi=0.0,
    )
    assert engine.position is not None
    assert engine.position.breakeven_done is True
    assert float(engine.position.stop_price) > entry_price


def test_chandelier_trail_ratchets_up() -> None:
    engine = machine(fast_params(trail_trigger_atr=1.5, trail_atr_multiple=2.0))
    entry_price = _open_long(engine)
    atr = engine.position.atr_points
    for index, peak in enumerate((1.6, 2.4, 3.2)):
        top = entry_price + peak * atr
        engine.on_bar(
            bar(24 + index, top, high=top, low=top - 0.3 * atr),
            obi=0.0,
        )
    assert engine.position is not None
    assert engine.position.trail_active is True
    assert float(engine.position.stop_price) > entry_price


def test_session_close_flattens_position() -> None:
    params = fast_params(session_windows="00:00-23:59", trade_until="18:35")
    engine = machine(params)
    entry_price = _open_long(engine)
    close_ms = minute_ms(hour=18, minute=36)
    events = engine.on_bar(
        bar(0, entry_price + 0.5, high=entry_price + 0.6, low=entry_price + 0.4, start_ms=close_ms),
        obi=0.0,
    )
    reasons = [event.reason for event in events if event.kind == "exit"]
    assert BreakoutExitReason.SESSION_CLOSE in reasons


def test_two_stops_arm_daily_cooldown() -> None:
    params = fast_params(max_consecutive_stops=1, cooldown_minutes=45)
    engine = machine(params)
    entry_price = _open_long(engine)
    engine.on_bar(bar(20, entry_price - 5.0, high=entry_price, low=entry_price - 6.0), obi=0.0)
    assert engine.position is None
    assert engine._consecutive_stops == 1  # noqa: SLF001
    # Повторный вход сразу после стопа запрещён кулдауном.
    events: list = []
    for item in _long_breakout_bars():
        for event in events:
            item.time_ms = event.time_ms  # noqa: F841 - оставляем бары как есть
        events.extend(engine.on_bar(item, obi=0.5))
    assert not [event for event in events if event.kind == "entry"]
    assert engine.counters.cooldown_skips >= 1


# --------------------------------------------------------------------- бэктестер


def test_backtester_sizing_respects_risk_and_margin() -> None:
    meta = make_meta("SR", price_step="1", step_amount="1", margin="5000")
    backtester = BreakoutBacktester(
        meta,
        fast_params(),
        BreakoutSettings(capital_rub=50_000, risk_pct_per_trade=1.0),
    )
    # Риск 1 % = 500 RUB; стоп 50 пунктов по 1 RUB -> 10 контрактов, но ГО даёт 8.
    assert backtester.size(50.0) == 8
    # Стоп 10 пунктов -> по риску 50, по ГО 8.
    assert backtester.size(10.0) == 8


def test_backtester_round_trip_fee_is_premium_per_side() -> None:
    meta = make_meta("SR", price_step="1", step_amount="1", margin="5000")
    backtester = BreakoutBacktester(
        meta, fast_params(), BreakoutSettings(capital_rub=50_000, tariff=Tariff.PREMIUM)
    )
    fee = backtester._round_trip_fee(dec(1000), 2)  # noqa: SLF001
    # Premium tier 1: 0.025 % за сторону, номинал 2 * 1000 RUB.
    assert float(fee) == pytest.approx(1.0, rel=1e-6)


def test_backtester_produces_trades_with_slippage_and_fees() -> None:
    meta = make_meta("SR", price_step="0.01", step_amount="0.01", margin="1000")
    bars = _long_breakout_bars(20)
    # после входа ведём цену вверх, чтобы сработал трейлинг/брейк-ивен
    entry_price = float(bars[-1].close)
    for index in range(1, 12):
        price = entry_price + index * 0.8
        bars.append(bar(20 + index, price, high=price + 0.1, low=price - 0.3, volume=1200))
    candles = [
        __import__("bot_a.backtest.candles", fromlist=["Candle"]).Candle(
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
        meta, fast_params(), BreakoutSettings(capital_rub=50_000, risk_pct_per_trade=1.0)
    )
    result = backtester.run(candles)
    assert isinstance(result, BreakoutResult)
    assert result.trades, "ожидалась хотя бы одна сделка"
    trade = result.trades[0]
    assert trade.quantity >= 1
    assert trade.fees_rub > 0
    assert trade.net_rub == pytest.approx(trade.gross_rub - trade.fees_rub, rel=1e-9)
    # Проскальзывание 1 шаг = 0.01 RUB на контракт против сделки.
    assert float(trade.entry_price) == pytest.approx(entry_price + 0.01)
    summary = result.summary(50_000)
    assert summary["trades"] == len(result.trades)
    assert summary["maxDrawdownRub"] >= 0.0


def test_result_metrics_profit_factor_payoff_and_drawdown() -> None:
    trades = [
        BreakoutTrade(
            ticker="X",
            side=BreakoutSide.LONG,
            quantity=1,
            entry_ms=1,
            exit_ms=2,
            entry_price=dec(100),
            exit_price=dec(110),
            gross_rub=10.0,
            fees_rub=1.0,
            net_rub=9.0,
            reason=BreakoutExitReason.TRAIL_STOP,
            atr_points=1.0,
            mfe_points=11.0,
            hold_minutes=5.0,
        ),
        BreakoutTrade(
            ticker="X",
            side=BreakoutSide.LONG,
            quantity=1,
            entry_ms=3,
            exit_ms=4,
            entry_price=dec(100),
            exit_price=dec(97),
            gross_rub=-3.0,
            fees_rub=1.0,
            net_rub=-4.0,
            reason=BreakoutExitReason.HARD_STOP,
            atr_points=1.0,
            mfe_points=0.2,
            hold_minutes=3.0,
        ),
    ]
    result = BreakoutResult(ticker="X", trades=trades)
    assert result.net_rub == pytest.approx(5.0)
    assert result.profit_factor == pytest.approx(9.0 / 4.0)
    assert result.payoff == pytest.approx(9.0 / 4.0)
    assert result.win_rate == pytest.approx(0.5)
    # Сделки упорядочены по выходу: сначала +9, затем -4 → просадка от пика 4 RUB.
    assert result.max_drawdown_rub == pytest.approx(4.0)
    assert result.max_drawdown_pct(50_000) == pytest.approx(4.0 / 500.0)


# --------------------------------------------------------------------- конфиг


def test_config_parses_breakout_presets() -> None:
    config = AppConfig.from_mapping(
        {
            "strategyMode": "breakout",
            "activeBreakout": "CNYRUBF",
            "breakoutPresets": {
                "CNYRUBF": {
                    "donchianBars": 45,
                    "trailAtrMultiple": 2.8,
                    "maxSqueezeRatio": 1.35,
                    "sessionWindows": "10:15-13:45",
                }
            },
        }
    )
    assert config.is_breakout() is True
    assert config.breakout_names() == ["CNYRUBF"]
    params = config.breakout()
    assert params.donchian_bars == 45
    assert params.trail_atr_multiple == pytest.approx(2.8)
    assert params.max_squeeze_ratio == pytest.approx(1.35)
    assert parse_windows(params.session_windows) == ((615, 825),)


def test_config_defaults_to_pairs_strategy() -> None:
    config = AppConfig.from_mapping({})
    assert config.is_breakout() is False
    assert config.breakout_names() == []
    assert isinstance(config.breakout(), BreakoutParams)


# --------------------------------------------------------------------- воркер


def test_bar_builder_aggregates_microprice_minutes() -> None:
    builder = BreakoutBarBuilder()
    start = minute_ms(hour=11, minute=0)
    assert builder.update(dec(100), 5, start) is None
    assert builder.update(dec(101), 5, start + 10_000) is None
    completed = builder.update(dec(99), 5, start + 59_000)
    assert completed is None  # бар ещё не закрыт
    finished = builder.update(dec(102), 5, start + 60_000)
    assert finished is not None
    assert float(finished.open) == pytest.approx(100.0)
    assert float(finished.high) == pytest.approx(101.0)
    assert float(finished.low) == pytest.approx(99.0)
    assert float(finished.close) == pytest.approx(99.0)
    assert finished.volume == 15


def test_live_worker_opens_and_closes_position(tmp_path) -> None:
    meta = make_meta("TEST", price_step="0.01", step_amount="0.01", margin="100")
    worker = BreakoutLiveWorker(
        meta,
        fast_params(),
        BreakoutSettings(capital_rub=100_000, risk_pct_per_trade=1.0),
        dry_run=True,
        journal_path=tmp_path / "journal.jsonl",
    )
    worker.warmup([])
    start = minute_ms(hour=11, minute=0)
    # Скармливаем стаканы: спокойный канал, затем пробой вверх и рост.
    prices = [100.0] * 12 + [100.8] * 14
    for index, price in enumerate(prices):
        book = make_book(price, start + index * 60_000, obi=0.9, uid="uid-TEST")
        worker.on_book(book)
    assert worker.state.bars >= 12
    summary = worker.summary()
    assert summary["ticker"] == "TEST"
    assert "net" in summary
    assert (tmp_path / "journal.jsonl").exists() or worker.state.trades == 0
