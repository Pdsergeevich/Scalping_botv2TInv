"""Синтез внутриминутных тиков и стаканов из свечей.

Для каждой минутной свечи строится детерминированный «мостик» цены
``O -> L -> H -> C`` (или ``O -> H -> L -> C``), 12 подтиков по 5 секунд,
top-5 стакан вокруг шага цены и лента сделок с объёмом, распределённым
по U-образному профилю.
"""

from __future__ import annotations

from decimal import Decimal
from typing import List, Optional

from bot_a.backtest.candles import Candle
from bot_a.math_engine.pair_spread import obi_weights
from bot_a.models.domain import (
    InstrumentMetadata,
    MarketEvent,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
    TradePrint,
)
from bot_a.utils.quotation import round_to_increment

TICKS_PER_MINUTE = 12
ZERO = Decimal("0")


def _volume_profile(count: int) -> List[float]:
    """U-образный профиль активности внутри минуты."""
    if count <= 1:
        return [1.0]
    half = (count - 1) / 2.0
    return [1.0 + 1.2 * abs(index - half) / half for index in range(count)]


def bridge_prices(candle: Candle, ticks: int = TICKS_PER_MINUTE, up_first: Optional[bool] = None) -> List[Decimal]:
    """Детерминированный путь цены внутри свечи.

    ``up_first`` позволяет зафиксировать порядок экстремумов по ведущей ноге
    (``None`` — по самой свече). Это устраняет фазовую инверсию внутри минуты.
    """
    ticks = max(int(ticks), 3)
    segments = max(1, ticks // 3)
    if up_first is None:
        up_first = candle.close >= candle.open
    anchors = (
        [candle.open, candle.low, candle.high, candle.close]
        if up_first
        else [candle.open, candle.high, candle.low, candle.close]
    )
    prices: List[Decimal] = []
    for index in range(3):
        start = anchors[index]
        end = anchors[index + 1]
        for step in range(1, segments + 1):
            ratio = Decimal(step) / Decimal(segments)
            prices.append(start + (end - start) * ratio)
    while len(prices) < ticks:
        prices.append(candle.close)
    return prices[:ticks]


def _quantities(total: float, depth: int) -> List[int]:
    weights = obi_weights(depth)
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return [max(int(total), 1)] * depth
    return [max(int(round(total * weight / weight_sum)), 1) for weight in weights]


def synthesize_book(
    meta: InstrumentMetadata,
    price: Decimal,
    volume_slice: int,
    obi_target: float,
    time_ms: int,
    depth: int = 5,
) -> OrderBookSnapshot:
    """Собирает top-``depth`` стакан вокруг ``price`` с заданным дисбалансом."""
    increment = meta.min_price_increment if meta.min_price_increment > 0 else Decimal("0.01")
    best_bid = round_to_increment(price, increment)
    best_ask = best_bid + increment
    base_volume = max(float(volume_slice) / max(depth * 2, 1), 1.0)
    total = base_volume * depth * 2.0
    ratio = min(max(obi_target, -0.95), 0.95)
    bid_volume = total * (1.0 + ratio) / 2.0
    ask_volume = total * (1.0 - ratio) / 2.0
    bids = tuple(
        OrderBookLevel(price=best_bid - increment * index, quantity=quantity)
        for index, quantity in enumerate(_quantities(bid_volume, depth))
    )
    asks = tuple(
        OrderBookLevel(price=best_ask + increment * index, quantity=quantity)
        for index, quantity in enumerate(_quantities(ask_volume, depth))
    )
    return OrderBookSnapshot(
        instrument_uid=meta.instrument_uid,
        time_ms=time_ms,
        bids=bids,
        asks=asks,
        last_price=best_bid,
    )


def synthesize_minute(
    leg: int,
    meta: InstrumentMetadata,
    candle: Candle,
    ticks_per_minute: int = TICKS_PER_MINUTE,
    depth: int = 5,
    up_first: Optional[bool] = None,
) -> List[MarketEvent]:
    """Превращает одну свечу в последовательность ``MarketEvent`` (стакан + принт)."""
    ticks = max(int(ticks_per_minute), 3)
    interval_ms = max(60_000 // ticks, 1)
    prices = bridge_prices(candle, ticks, up_first)
    weights = _volume_profile(ticks)
    weight_sum = sum(weights)
    body_imbalance = 0.0
    span = float(candle.high - candle.low)
    if span > 0:
        body_imbalance = float(candle.close - candle.open) / span
    body_imbalance = max(min(body_imbalance, 1.0), -1.0)
    events: List[MarketEvent] = []
    previous_price = candle.open
    for index, price in enumerate(prices):
        time_ms = candle.time_ms + (index + 1) * interval_ms
        direction = 0.0
        if price > previous_price:
            direction = 1.0
        elif price < previous_price:
            direction = -1.0
        obi_target = max(min(0.45 * body_imbalance + 0.45 * direction, 0.95), -0.95)
        volume_slice = max(int(candle.volume * weights[index] / weight_sum), 1)
        book = synthesize_book(meta, price, volume_slice, obi_target, time_ms, depth)
        events.append(MarketEvent.of_book(leg, book))
        side = Side.BUY if direction >= 0 else Side.SELL
        events.append(
            MarketEvent.of_trade(
                leg,
                TradePrint(
                    instrument_uid=meta.instrument_uid,
                    time_ms=time_ms,
                    price=book.bids[0].price,
                    quantity=volume_slice,
                    side=side,
                ),
            )
        )
        previous_price = price
    return events

def synthesize_pair_minute(
    leg1: int,
    meta1: InstrumentMetadata,
    candle1: Optional[Candle],
    leg2: int,
    meta2: InstrumentMetadata,
    candle2: Optional[Candle],
    ticks_per_minute: int = TICKS_PER_MINUTE,
    depth: int = 5,
) -> List[MarketEvent]:
    """Фазово-синхронная пара тиков внутри одной минуты.

    Порядок экстремумов (``O->L->H->C`` или ``O->H->L->C``) задаётся **ведущей ногой
    (Leg2)** и применяется к обеим ногам. Это убирает фазовую инверсию, когда
    Leg1 стоит на максимуме, а Leg2 — на минимуме внутри одного бара.
    """
    phase: Optional[bool] = None
    if candle2 is not None:
        phase = candle2.close >= candle2.open
    elif candle1 is not None:
        phase = candle1.close >= candle1.open
    group: List[MarketEvent] = []
    if candle1 is not None:
        group.extend(synthesize_minute(leg1, meta1, candle1, ticks_per_minute, depth, up_first=phase))
    if candle2 is not None:
        group.extend(synthesize_minute(leg2, meta2, candle2, ticks_per_minute, depth, up_first=phase))
    group.sort(key=lambda event: (event.time_ms, 0 if event.is_book else 1))
    return group


def synthesize_close_minute(
    leg: int,
    meta: InstrumentMetadata,
    candle: Candle,
    depth: int = 5,
) -> List[MarketEvent]:
    """Zero-artifact режим: только закрытия минуты, без синтетики."""
    time_ms = candle.time_ms + 60_000
    span = float(candle.high - candle.low)
    imbalance = 0.0
    if span > 0:
        imbalance = max(min(float(candle.close - candle.open) / span, 1.0), -1.0)
    volume = max(int(candle.volume), 1)
    book = synthesize_book(meta, candle.close, volume, 0.45 * imbalance, time_ms, depth)
    side = Side.BUY if imbalance >= 0 else Side.SELL
    return [
        MarketEvent.of_book(leg, book),
        MarketEvent.of_trade(
            leg,
            TradePrint(
                instrument_uid=meta.instrument_uid,
                time_ms=time_ms,
                price=candle.close,
                quantity=volume,
                side=side,
            ),
        ),
    ]
