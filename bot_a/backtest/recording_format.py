"""Формат записи рыночных данных (JSONL, по одной строке на событие).

Схема строки::

    {"t": 1690000000123, "k": "book", "leg": 1, "uid": "...",
     "b": [[1.2345, 10], ...], "a": [[1.2346, 12], ...], "lp": 1.2345}
    {"t": 1690000000123, "k": "trade", "leg": 2, "uid": "...",
     "p": 1.2345, "q": 5, "s": "buy"}
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from bot_a.models.domain import (
    MarketEvent,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
    TradePrint,
)
from bot_a.utils.quotation import dec


def event_to_dict(event: MarketEvent) -> Dict[str, Any]:
    if event.is_book and event.book is not None:
        book = event.book
        return {
            "t": book.time_ms,
            "k": "book",
            "leg": event.leg,
            "uid": book.instrument_uid,
            "b": [[str(level.price), level.quantity] for level in book.bids],
            "a": [[str(level.price), level.quantity] for level in book.asks],
            "lp": str(book.last_price) if book.last_price is not None else None,
        }
    trade = event.trade
    if trade is None:
        raise ValueError("MarketEvent без стакана и без сделки")
    return {
        "t": trade.time_ms,
        "k": "trade",
        "leg": event.leg,
        "uid": trade.instrument_uid,
        "p": str(trade.price),
        "q": trade.quantity,
        "s": trade.side.value if trade.side is not None else None,
    }



def event_from_dict(data: Dict[str, Any]) -> Optional[MarketEvent]:
    kind = data.get("k")
    leg = int(data.get("leg", 0) or 0)
    time_ms = int(data.get("t", 0) or 0)
    uid = str(data.get("uid", "") or "")
    if kind == "book":
        bids = tuple(
            OrderBookLevel(price=dec(price), quantity=int(quantity))
            for price, quantity in (data.get("b") or [])
        )
        asks = tuple(
            OrderBookLevel(price=dec(price), quantity=int(quantity))
            for price, quantity in (data.get("a") or [])
        )
        last_price = data.get("lp")
        book = OrderBookSnapshot(
            instrument_uid=uid,
            time_ms=time_ms,
            bids=bids,
            asks=asks,
            last_price=dec(last_price) if last_price is not None else None,
        )
        return MarketEvent.of_book(leg, book)
    if kind == "trade":
        side_raw = data.get("s")
        side = Side(side_raw) if side_raw in (Side.BUY.value, Side.SELL.value) else None
        trade = TradePrint(
            instrument_uid=uid,
            time_ms=time_ms,
            price=dec(data.get("p")),
            quantity=int(data.get("q", 0) or 0),
            side=side,
        )
        return MarketEvent.of_trade(leg, trade)
    return None


def event_from_json(line: str) -> Optional[MarketEvent]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    return event_from_dict(json.loads(stripped))


def levels_from_jsonb(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, str):
        value = json.loads(value)
    return tuple(OrderBookLevel(price=dec(price), quantity=int(quantity)) for price, quantity in value)


def book_from_row(row: tuple) -> MarketEvent:
    """Строка PostgreSQL: ``(ts_ms, instrument_uid, leg, bids, asks, last_price)``."""
    ts_ms, instrument_uid, leg, bids, asks, last_price = row
    book = OrderBookSnapshot(
        instrument_uid=str(instrument_uid),
        time_ms=int(ts_ms),
        bids=levels_from_jsonb(bids),
        asks=levels_from_jsonb(asks),
        last_price=dec(last_price) if last_price is not None else None,
    )
    return MarketEvent.of_book(int(leg), book)


def trade_from_row(row: tuple) -> MarketEvent:
    """Строка PostgreSQL: ``(ts_ms, instrument_uid, leg, price, quantity, side)``."""
    ts_ms, instrument_uid, leg, price, quantity, side_raw = row
    side = None
    if side_raw in (Side.BUY.value, Side.SELL.value):
        side = Side(side_raw)
    trade = TradePrint(
        instrument_uid=str(instrument_uid),
        time_ms=int(ts_ms),
        price=dec(price),
        quantity=int(quantity),
        side=side,
    )
    return MarketEvent.of_trade(int(leg), trade)
