"""Тесты v3.2: фильтр «пилы», частичная фиксация, общий счёт и мульти-тикерный раннер.

Проверяем четыре блока:

1. **Индикаторы**: Kaufman ER (тренд → 1, пила → 0) и наклон EMA90 в единицах ATR.
2. **Фильтр входа**: ER/наклон блокируют пробои в боковике и не мешают тренду.
3. **Выходы**: «мягкий» стоп (Tier-1) и частичная фиксация (Tier-2) — доля, стоп, учёт сделок.
4. **Счёт и раннер**: сайзинг от общего капитала, лимит утилизации ГО, детерминизм,
   сохранение/восстановление позиций в `state/swing_positions.json`.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal

import pytest

from bot_a.backtest.breakout_backtest import BreakoutBacktester, BreakoutSettings
from bot_a.backtest.candles import Candle
from bot_a.backtest.swing_portfolio import SwingAccountBacktester, SwingAccountSettings
from bot_a.config import AppConfig
from bot_a.engine.breakout_fsm import (
    Bar,
    BreakoutExitReason,
    BreakoutIndicators,
    BreakoutParams,
    BreakoutSide,
    MomentumBreakoutStateMachine,
)
from bot_a.services.breakout_worker import BreakoutBarBuilder
from bot_a.services.swing_runner import BasketPosition, SwingBasketRunner, SwingBasketSettings
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import MSK, msk_to_ms
from tests.helpers import make_book, make_meta


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


def v32_params(**overrides) -> BreakoutParams:
    base = dict(
        donchian_bars=10,
        atr_bars=5,
        squeeze_median_bars=5,
        max_squeeze_ratio=0.0,
        max_relative_squeeze=0.0,
        ema_fast_bars=2,
        ema_slow_bars=3,
        volume_lookback=5,
        volume_multiple=0.5,
        close_in_range=0.3,
        min_atr_to_fee_ratio=0.0,
        obi_long_threshold=0.2,
        obi_short_threshold=-0.2,
        breakout_buffer_steps=1,
        session_windows="00:00-23:59",
        trade_until="18:35",
        initial_stop_atr_multiple=2.0,
        min_stop_steps=1,
        failed_breakout_minutes=0,
        failed_breakout_bars=3,
        trail_trigger_atr=2.5,
        trail_atr_multiple=4.0,
        take_profit_atr=0.0,
        max_consecutive_stops=5,
        cooldown_minutes=0,
        slippage_steps=1,
    )
    base.update(overrides)
    return BreakoutParams(**base)


def machine(params: BreakoutParams, ticker: str = "TEST") -> MomentumBreakoutStateMachine:
    return MomentumBreakoutStateMachine(
        make_meta(ticker, price_step="0.01", step_amount="0.01"), params
    )


# --------------------------------------------------------------- индикаторы
def test_efficiency_ratio_trend_vs_chop() -> None:
    params = v32_params(efficiency_ratio_bars=10)
    trend = BreakoutIndicators(params, Decimal(1))
    snapshot = None
    for index in range(12):
        snapshot = trend.update(bar(index, 100.0 + index))
    assert snapshot.efficiency_ratio == pytest.approx(1.0, abs=1e-6)

    chop = BreakoutIndicators(params, Decimal(1))
    for index in range(12):
        snapshot = chop.update(bar(index, 100.0 + (0.5 if index % 2 else -0.5)))
    assert snapshot.efficiency_ratio < 0.15


def test_ema_slope_atr_is_flat_in_chop_and_positive_in_trend() -> None:
    params = v32_params(efficiency_ratio_bars=10)
    trend = BreakoutIndicators(params, Decimal(1))
    snapshot = None
    for index in range(14):
        snapshot = trend.update(bar(index, 100.0 + index * 0.5, high=100.6 + index * 0.5))
    assert snapshot.ema_slope_atr > 2.0

    chop = BreakoutIndicators(params, Decimal(1))
    for index in range(14):
        snapshot = chop.update(bar(index, 100.0 + (0.2 if index % 2 else -0.2), high=100.3, low=99.7))
    assert snapshot.ema_slope_atr < 0.5


# --------------------------------------------------------------- фильтр входа
def _breakout_bars(count: int = 24, rising: bool = True) -> list[Bar]:
    bars: list[Bar] = []
    for index in range(count):
        close = 100.0 + index * 0.05 if rising else 100.0 + (0.3 if index % 2 else -0.3)
        bars.append(bar(index, close, high=close + 0.05, low=close - 0.15, volume=1000))
    last = float(bars[-1].close)
    bars.append(bar(count, last + 0.6, high=last + 0.65, low=last + 0.2, open_=last + 0.2, volume=5000))
    return bars


def _chop_bars(count: int = 24) -> list[Bar]:
    """Боковик: цена пилит ±0.3 вокруг медленно растущей середины, затем пробой."""
    bars: list[Bar] = []
    for index in range(count):
        mid = 100.0 + index * 0.02
        close = mid + (0.3 if index % 2 else -0.3)
        bars.append(bar(index, close, high=close + 0.05, low=close - 0.15, volume=1000))
    last = float(bars[-1].close)
    mid = 100.0 + count * 0.02
    top = max(float(item.high) for item in bars)
    bars.append(
        bar(count, max(top + 0.6, mid + 0.3), high=max(top + 0.6, mid + 0.3) + 0.05, low=last, open_=last, volume=5000)
    )
    return bars


def _trend_engine(**overrides) -> MomentumBreakoutStateMachine:
    params = v32_params(**overrides)
    engine = machine(params)
    # Индикаторы должны «увидеть» тренд до бара пробоя.
    for item in _breakout_bars()[:-1]:
        engine.on_bar(item)
    return engine


def test_efficiency_ratio_filter_blocks_chop_entries() -> None:
    bars = _chop_bars()
    gated = machine(v32_params(min_efficiency_ratio=0.35))
    events = []
    for item in bars:
        events.extend(gated.on_bar(item))
    assert not [event for event in events if event.kind == "entry"]
    assert gated.counters.chop_skips >= 1
    assert gated.position is None

    # Тот же ряд без фильтра — вход есть (иначе тест ничего не доказывает).
    free = machine(v32_params())
    free_events = []
    for item in bars:
        free_events.extend(free.on_bar(item))
    assert [event for event in free_events if event.kind == "entry"]


def test_efficiency_ratio_filter_allows_trend_entries() -> None:
    engine = _trend_engine(min_efficiency_ratio=0.35)
    # В тренде ER ≈ 1 — тот же фильтр не мешает.
    assert engine._indicators.snapshot.efficiency_ratio > 0.5  # noqa: SLF001
    events = engine.on_bar(_breakout_bars()[-1])
    assert [event for event in events if event.kind == "entry"]


def test_ema_slope_filter_blocks_flat_ma() -> None:
    params = v32_params(min_ema_slope_atr=5.0)  # недостижимый наклон
    engine = machine(params)
    events = []
    for item in _breakout_bars():
        events.extend(engine.on_bar(item))
    assert not [event for event in events if event.kind == "entry"]
    assert engine.counters.chop_skips >= 1


def test_chop_filters_disabled_by_default() -> None:
    params = v32_params()
    assert params.min_efficiency_ratio == 0.0
    assert params.min_ema_slope_atr == 0.0
    engine = machine(params)
    events = []
    for item in _breakout_bars():
        events.extend(engine.on_bar(item))
    assert engine.counters.chop_skips == 0
    assert [event for event in events if event.kind == "entry"]


# --------------------------------------------------------------- мягкий стоп (Tier-1)
def test_soft_stop_prevents_full_loss_after_partial_profit() -> None:
    params = v32_params(
        soft_breakeven_trigger_atr=1.0,
        soft_stop_atr=0.0,
        initial_stop_atr_multiple=2.0,
        breakeven_trigger_atr=3.0,  # Tier-2 не мешает проверить Tier-1
    )
    engine = machine(params)
    for item in _breakout_bars():
        engine.on_bar(item)
    assert engine.position is not None
    entry_price = float(engine.position.entry_price)
    atr = engine.position.atr_points
    # Прибыль 1.2 ATR: экстремум учитывается после ведения, поэтому стоп
    # подтягивается со следующего бара.
    engine.on_bar(
        bar(30, entry_price + 1.2 * atr, high=entry_price + 1.25 * atr, low=entry_price + 0.9 * atr)
    )
    assert engine.position is not None
    engine.on_bar(
        bar(31, entry_price + 1.15 * atr, high=entry_price + 1.2 * atr, low=entry_price + 1.0 * atr)
    )
    assert engine.position is not None
    assert engine.position.soft_stop_done is True
    # Стоп подтянут минимум к безубытку (может быть выше из-за breakeven-ратчета).
    assert float(engine.position.stop_price) >= entry_price - 0.02
    # Разворот к входу — выход по SoftStop, а не полный −2 ATR.
    events = engine.on_bar(
        bar(32, entry_price - 0.1, high=entry_price + 0.05, low=entry_price - 0.2)
    )
    reasons = [event.reason for event in events if event.kind == "exit"]
    assert BreakoutExitReason.SOFT_STOP in reasons
    assert engine.position is None
    exit_event = next(event for event in events if event.kind == "exit")
    # Гэп через подтянутый стоп: убыток ограничен ~1 ATR вместо полного −2 ATR.
    assert float(exit_event.price) >= entry_price - atr


def test_soft_stop_far_when_mfe_small() -> None:
    params = v32_params(soft_breakeven_trigger_atr=2.0, soft_stop_atr=0.0)
    engine = machine(params)
    for item in _breakout_bars():
        engine.on_bar(item)
    entry_price = float(engine.position.entry_price)
    atr = engine.position.atr_points
    engine.on_bar(
        bar(30, entry_price + 1.0 * atr, high=entry_price + 1.1 * atr, low=entry_price + 0.5 * atr)
    )
    engine.on_bar(
        bar(31, entry_price + 1.0 * atr, high=entry_price + 1.1 * atr, low=entry_price + 0.6 * atr)
    )
    assert engine.position is not None
    assert engine.position.soft_stop_done is False
    assert float(engine.position.stop_price) < entry_price - atr


# --------------------------------------------------------------- частичная фиксация (Tier-2)
def test_scale_out_emits_partial_event_and_keeps_position() -> None:
    params = v32_params(scale_out_atr=1.0, scale_out_fraction=0.5, scale_out_stop_atr=0.0)
    engine = machine(params)
    for item in _breakout_bars():
        engine.on_bar(item)
    engine.position.quantity = 10
    entry_price = float(engine.position.entry_price)
    atr = engine.position.atr_points
    engine.on_bar(
        bar(30, entry_price + 1.2 * atr, high=entry_price + 1.3 * atr, low=entry_price + 0.9 * atr)
    )
    events = engine.on_bar(
        bar(31, entry_price + 1.25 * atr, high=entry_price + 1.35 * atr, low=entry_price + 1.0 * atr)
    )
    partials = [event for event in events if event.kind == "partial"]
    assert len(partials) == 1
    assert partials[0].quantity == 5
    assert partials[0].reason is BreakoutExitReason.PARTIAL
    assert engine.position is not None
    assert engine.position.quantity == 5
    assert engine.position.scale_out_done is True
    assert float(engine.position.stop_price) >= entry_price - 0.01


def test_scale_out_through_backtester_records_two_trades() -> None:
    meta = make_meta("SR", price_step="0.01", step_amount="0.01", margin="1000")
    params = v32_params(scale_out_atr=1.0, scale_out_fraction=0.5, scale_out_stop_atr=0.5)
    bars = _breakout_bars(20)
    price = float(bars[-1].close)
    for index in range(1, 8):
        level = price + index * 0.4
        bars.append(bar(20 + index, level, high=level + 0.05, low=level - 0.1, volume=1200))
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
    result = BreakoutBacktester(
        meta, params, BreakoutSettings(capital_rub=50_000, risk_pct_per_trade=1.0)
    ).run(candles)
    assert result.trades
    reasons = {trade.reason for trade in result.trades}
    assert BreakoutExitReason.PARTIAL in reasons
    partial = next(trade for trade in result.trades if trade.reason is BreakoutExitReason.PARTIAL)
    assert partial.quantity >= 1


# --------------------------------------------------------------- общий счёт
def _candles(count: int = 200, step_minutes: int = 30, start_price: float = 100.0) -> list[Candle]:
    result: list[Candle] = []
    start = minute_ms(hour=10)
    for index in range(count):
        close = start_price + index * 0.05
        result.append(
            Candle(
                time_ms=start + index * step_minutes * 60_000,
                open=dec(close - 0.02),
                high=dec(close + 0.05),
                low=dec(close - 0.15),
                close=dec(close),
                volume=1000,
            )
        )
    return result


def test_shared_account_uses_total_equity_not_slices() -> None:
    meta_a = make_meta("AAA", price_step="0.01", step_amount="0.01", margin="1000")
    params = v32_params()
    engine = SwingAccountBacktester(
        [(meta_a, params)],
        SwingAccountSettings(
            capital_rub=50_000, risk_pct_per_trade=1.0, max_margin_utilization_pct=60
        ),
    )
    # Риск 1 % от 50 000 = 500 ₽; стоп 2 ATR ≈ 4 пункта -> 125 контрактов риска,
    # но ГО (1000 × 1.25) и лимит 60 % (30 000 ₽) дают 24 контракта.
    quantity = engine.size(meta_a, 4.0)
    assert quantity == 24


def test_shared_account_rejects_when_margin_exhausted() -> None:
    meta_exp = make_meta("EXP", price_step="1", step_amount="1", margin="20000")
    params = v32_params()
    engine = SwingAccountBacktester(
        [(meta_exp, params)],
        SwingAccountSettings(
            capital_rub=50_000, risk_pct_per_trade=5.0, max_margin_utilization_pct=60
        ),
    )
    assert engine.size(meta_exp, 1.0) == 1  # 30 000 / 25 000 = 1
    assert engine.size(meta_exp, 1.0, equity=20_000) == 0  # 12 000 / 25 000 = 0
    # Занятое ГО не даёт открыть вторую позицию на том же счёте.
    assert engine.size(meta_exp, 1.0, margin_used=25_000) == 0


def test_shared_account_is_deterministic_on_dict_order() -> None:
    params = v32_params(min_efficiency_ratio=0.0)
    metas = {
        "AAA": make_meta("AAA", price_step="0.01", step_amount="0.01", margin="1000"),
        "BBB": make_meta("BBB", price_step="0.01", step_amount="0.01", margin="1000"),
    }
    data_a = {"AAA": _candles(), "BBB": _candles(start_price=50.0)}
    data_b = {"BBB": _candles(start_price=50.0), "AAA": _candles()}
    settings = SwingAccountSettings(
        capital_rub=50_000, risk_pct_per_trade=1.0, max_margin_utilization_pct=60
    )
    first = SwingAccountBacktester(
        [(metas["AAA"], params), (metas["BBB"], params)], settings
    ).run(data_a)
    second = SwingAccountBacktester(
        [(metas["BBB"], params), (metas["AAA"], params)], settings
    ).run(data_b)
    assert first.net_rub == pytest.approx(second.net_rub)
    assert len(first.trades) == len(second.trades)


def test_shared_account_equity_grows_with_realized_pnl() -> None:
    meta = make_meta("AAA", price_step="0.01", step_amount="0.01", margin="1000")
    params = v32_params()
    engine = SwingAccountBacktester(
        [(meta, params)],
        SwingAccountSettings(capital_rub=50_000, risk_pct_per_trade=1.0),
    )
    result = engine.run({"AAA": _candles(120)})
    if result.trades:
        assert result.final_equity == pytest.approx(50_000 + result.net_rub, abs=0.01)


# --------------------------------------------------------------- мульти-тикерный раннер
def _runner(tmp_path, tickers=("AAA", "BBB"), **overrides) -> SwingBasketRunner:
    # В синтетических стаканах объём постоянен, поэтому дельта объёма = 0:
    # отключаем объёмный фильтр, он проверяется отдельно в тестах ФСМ.
    params = v32_params(volume_multiple=0.0, donchian_bars=8, atr_bars=5)
    instruments = [
        (make_meta(ticker, price_step="0.01", step_amount="0.01", margin="1000"), params)
        for ticker in tickers
    ]
    settings = SwingBasketSettings(
        capital_rub=50_000,
        risk_pct_per_trade=1.0,
        max_margin_utilization_pct=60,
        bar_minutes=1,
        state_path=tmp_path / "state" / "swing_positions.json",
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    return SwingBasketRunner(instruments, settings)


def _warm_bars(count: int = 15, base: float = 100.0) -> list[Bar]:
    """Ровная история для прогрева индикаторов перед live-стаканами."""
    bars: list[Bar] = []
    for index in range(count):
        close = base + (0.02 if index % 2 else -0.02)
        bars.append(
            Bar(
                time_ms=minute_ms(hour=10) + index * 60_000,
                open=dec(base),
                high=dec(base + 0.05),
                low=dec(base - 0.05),
                close=dec(close),
                volume=1000,
            )
        )
    return bars


def _feed_minute(
    runner: SwingBasketRunner, ticker: str, minute: int, mid: float, *, ascending: bool = False
) -> None:
    """Три стакана внутри минуты, чтобы у бара был реальный диапазон."""
    start = minute_ms(hour=10, minute=15) + minute * 60_000
    offsets = (0.0, -0.02, 0.02) if ascending else (0.0, 0.02, -0.02)
    for offset_index, delta in enumerate(offsets):
        runner.on_book(
            ticker,
            make_book(mid + delta, start + offset_index * 20_000, obi=0.9, uid=f"uid-{ticker}"),
        )


def _run_entry_scenario(runner: SwingBasketRunner) -> None:
    runner.warmup({"AAA": _warm_bars()})
    _feed_minute(runner, "AAA", 0, 100.0)
    _feed_minute(runner, "AAA", 1, 100.8, ascending=True)
    # Бар минуты 1 закрывается только на границе минуты 2 (бар эмитится завершённым).
    _feed_minute(runner, "AAA", 2, 100.8, ascending=True)


def test_runner_opens_position_from_books(tmp_path) -> None:
    runner = _runner(tmp_path)
    _run_entry_scenario(runner)
    assert "AAA" in runner._positions  # noqa: SLF001
    position = runner._positions["AAA"]  # noqa: SLF001
    assert position.quantity >= 1
    assert position.margin_rub > 0
    assert position.stop_price < position.entry_price  # LONG: стоп ниже входа
    runner.save_state()
    payload = json.loads((tmp_path / "state" / "swing_positions.json").read_text(encoding="utf-8"))
    assert "AAA" in payload["positions"]
    assert payload["equity"] == pytest.approx(50_000)
    assert payload["marginUsed"] == pytest.approx(position.margin_rub)


def test_runner_state_survives_restart(tmp_path) -> None:
    runner = _runner(tmp_path)
    _run_entry_scenario(runner)
    assert "AAA" in runner._positions  # noqa: SLF001
    runner.save_state()

    restarted = _runner(tmp_path)
    restarted.warmup({"AAA": _warm_bars()})
    assert restarted.restore() is True
    position = restarted._positions["AAA"]  # noqa: SLF001
    assert position.quantity >= 1
    # ФСМ получил восстановленную позицию — стоп и трейл продолжат работать.
    machine = restarted._machines["AAA"]  # noqa: SLF001
    assert machine.position is not None
    assert machine.position.stop_price == position.stop_price
    assert machine.position.entry_price == position.entry_price
    assert restarted.margin_used == pytest.approx(position.margin_rub)


def test_runner_restore_without_state_is_safe(tmp_path) -> None:
    runner = _runner(tmp_path)
    assert runner.restore() is False
    assert runner.equity == pytest.approx(50_000)


def test_basket_position_roundtrip() -> None:
    position = BasketPosition(
        ticker="AAA",
        side=BreakoutSide.LONG,
        quantity=3,
        entry_price=dec("100.5"),
        entry_ms=1_700_000_000_000,
        stop_price=dec("98.0"),
        initial_stop=dec("98.0"),
        atr_points=1.25,
        margin_rub=1250.0,
        entry_fee=dec("1.5"),
        mfe_points=2.5,
        breakeven_done=True,
    )
    restored = BasketPosition.from_dict(position.to_dict())
    assert restored.ticker == position.ticker
    assert restored.side is position.side
    assert restored.entry_price == position.entry_price
    assert restored.breakeven_done is True
    assert restored.mfe_points == pytest.approx(2.5)


def test_runner_summary_shape(tmp_path) -> None:
    runner = _runner(tmp_path)
    runner.warmup({})
    summary = runner.summary()
    assert summary["mode"] == "ALL_SWING"
    assert summary["capital"] == pytest.approx(50_000)
    assert summary["trades"] == 0
    assert summary["openPositions"] == {}


# --------------------------------------------------------------- конфигурация
def test_config_parses_swing_basket() -> None:
    config = AppConfig.from_mapping(
        {
            "strategyMode": "breakout",
            "swingBasket": ["CNYRUBF", "GLDRUBF", "USDRUBF"],
            "breakoutPresets": {"CNYRUBF": {"donchianBars": 20}},
        }
    )
    assert config.swing_basket == ["CNYRUBF", "GLDRUBF", "USDRUBF"]
    assert config.swing_tickers() == ["CNYRUBF", "GLDRUBF", "USDRUBF"]


def test_config_swing_basket_defaults_to_presets() -> None:
    config = AppConfig.from_mapping(
        {"strategyMode": "breakout", "breakoutPresets": {"CNYRUBF": {}, "GLDRUBF": {}}}
    )
    assert config.swing_tickers() == ["CNYRUBF", "GLDRUBF"]


def test_bar_builder_respects_custom_timeframe() -> None:
    builder = BreakoutBarBuilder(minutes=30)
    start = minute_ms(hour=10)
    assert builder.update(dec(100), 10, start) is None
    assert builder.update(dec(101), 10, start + 20 * 60_000) is None
    completed = builder.update(dec(102), 10, start + 30 * 60_000)
    assert completed is not None
    assert float(completed.high) == pytest.approx(101.0)
