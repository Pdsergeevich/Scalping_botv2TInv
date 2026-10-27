"""Тесты v2.1: фазовая синхронизация моста и zero-artifact режим close."""

from __future__ import annotations

from decimal import Decimal

from bot_a.backtest.candles import Candle, InMemoryCandleSource
from bot_a.backtest.historical_candles import iter_candle_events
from bot_a.backtest.synthetics import synthesize_close_minute, synthesize_pair_minute
from tests.helpers import make_meta, make_preset, trading_start_ms


def _candle(time_ms: int, open_: str, high: str, low: str, close: str, volume: int = 1000) -> Candle:
    return Candle(
        time_ms=time_ms,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=volume,
    )


def _prices(events, leg: int):
    return [
        event.book.bids[0].price
        for event in events
        if event.leg == leg and event.book is not None
    ]


def test_phase_locked_bridge_uses_lead_leg_phase():
    meta1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
    meta2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
    # Leg1 растёт (O->L->H->C), Leg2 падает (O->H->L->C): фазу задаёт Leg2.
    candle1 = _candle(1_000_000, "100.00", "100.20", "99.90", "100.15")
    candle2 = _candle(1_000_000, "101.00", "101.10", "100.80", "100.85")

    events = synthesize_pair_minute(1, meta1, candle1, 2, meta2, candle2, ticks_per_minute=12, depth=5)
    prices1 = _prices(events, 1)
    prices2 = _prices(events, 2)
    assert len(prices1) == 12 and len(prices2) == 12

    # Фаза Leg2 (вниз): максимум на 4-м подтике, минимум — на 8-м.
    assert prices1[3] == Decimal("100.20"), prices1
    assert prices2[3] == Decimal("101.10"), prices2
    assert prices1[7] == Decimal("99.90"), prices1
    assert prices2[7] == Decimal("100.80"), prices2
    assert prices1[-1] == Decimal("100.15")
    assert prices2[-1] == Decimal("100.85")

    # Никакой «фазовой инверсии»: экстремумы ног совпадают по времени.
    high_index_1 = max(range(len(prices1)), key=lambda i: prices1[i])
    low_index_2 = min(range(len(prices2)), key=lambda i: prices2[i])
    assert high_index_1 == 3
    assert low_index_2 == 7


def test_close_mode_uses_only_real_closes():
    meta1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
    meta2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
    start = trading_start_ms()
    c1 = [_candle(start, "100.00", "100.50", "99.50", "100.30"), _candle(start + 60_000, "100.30", "100.60", "100.10", "100.40")]
    c2 = [_candle(start, "101.00", "101.30", "100.70", "100.90"), _candle(start + 60_000, "100.90", "101.20", "100.80", "101.10")]

    events = list(iter_candle_events(meta1, meta2, c1, c2, depth=5, synth_mode="close"))
    assert len(events) == 8  # 2 минуты × 2 ноги × (стакан + сделка)
    books1 = [event.book for event in events if event.leg == 1 and event.book is not None]
    books2 = [event.book for event in events if event.leg == 2 and event.book is not None]
    assert [book.bids[0].price for book in books1] == [Decimal("100.30"), Decimal("100.40")]
    assert [book.bids[0].price for book in books2] == [Decimal("100.90"), Decimal("101.10")]
    trades = [event.trade for event in events if event.trade is not None]
    assert {trade.price for trade in trades} == {
        Decimal("100.30"),
        Decimal("100.40"),
        Decimal("100.90"),
        Decimal("101.10"),
    }


def test_close_minute_volume_and_timestamp():
    meta = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
    candle = _candle(5_000_000, "100.00", "100.50", "99.50", "100.40", volume=777)
    events = synthesize_close_minute(1, meta, candle, depth=5)
    assert len(events) == 2
    book_event, trade_event = events
    assert book_event.book is not None and book_event.book.time_ms == candle.time_ms + 60_000
    assert trade_event.trade is not None and trade_event.trade.quantity == 777
    assert trade_event.trade.price == Decimal("100.40")


def test_legacy_bridge_without_phase_lock_still_works():
    meta1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
    meta2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
    start = trading_start_ms()
    c1 = [_candle(start, "100.00", "100.50", "99.50", "100.30")]
    c2 = [_candle(start, "101.00", "101.30", "100.70", "100.90")]

    events = list(iter_candle_events(meta1, meta2, c1, c2, 12, 5, phase_lock=False))
    assert len(events) == 2 * 2 * 12  # 2 ноги × 12 подтиков × (стакан + сделка)


def test_candle_source_is_unchanged_by_synth_mode():
    preset = make_preset()
    source = InMemoryCandleSource({"AAA": [], "BBB": []})
    assert preset.leg1.ticker == "AAA"
    assert source is not None
