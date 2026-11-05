"""Источник котировок MOEX ISS для paper/watch без токена T-Invest.

Даёт тот же интерфейс, что и `TInvestRestClient` (`get_order_book`), поэтому живой
воркер стратегии v3.0 не знает, откуда пришёл стакан. ISS обновляет `orderbook.json`
не чаще раза в секунду — для минутных баров этого достаточно.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from bot_a.models.domain import InstrumentMetadata, OrderBookLevel, OrderBookSnapshot
from bot_a.utils.quotation import dec

ISS_BASE = "https://iss.moex.com/iss"


def _url(meta: InstrumentMetadata, depth: int) -> str:
    if meta.is_futures:
        base = f"{ISS_BASE}/engines/futures/markets/forts/securities/{meta.ticker}"
    else:
        base = (
            f"{ISS_BASE}/engines/stock/markets/shares/boards/{meta.class_code}"
            f"/securities/{meta.ticker}"
        )
    return f"{base}/orderbook.json?depth={depth}&iss.meta=off"


def _rows(payload: Dict[str, Any]) -> list:
    block = payload.get("orderbook") or {}
    columns = block.get("columns") or []
    index = {name: position for position, name in enumerate(columns)}
    return [{name: row[position] for name, position in index.items()} for row in block.get("data") or []]


def _parse(meta: InstrumentMetadata, payload: Dict[str, Any], depth: int) -> Optional[OrderBookSnapshot]:
    bids = []
    asks = []
    stamp_ms = 0
    for row in _rows(payload):
        side = str(row.get("BUYSELL") or "").upper()
        price = dec(row.get("PRICE") or 0)
        quantity = int(row.get("QUANTITY") or 0)
        if price <= 0 or quantity <= 0:
            continue
        if side == "B":
            bids.append((price, quantity))
        elif side == "S":
            asks.append((price, quantity))
        moment = row.get("UPDATETIME")
        if moment:
            stamp_ms = _msk_stamp_to_ms(str(moment))
    if not bids and not asks:
        return None
    bids.sort(key=lambda item: item[0], reverse=True)
    asks.sort(key=lambda item: item[0])
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    return OrderBookSnapshot(
        instrument_uid=meta.instrument_uid or meta.ticker,
        time_ms=stamp_ms or now_ms,
        bids=tuple(OrderBookLevel(price=price, quantity=quantity) for price, quantity in bids[:depth]),
        asks=tuple(OrderBookLevel(price=price, quantity=quantity) for price, quantity in asks[:depth]),
    )


def _msk_stamp_to_ms(stamp: str) -> int:
    try:
        hour, minute, second = (int(part) for part in stamp.split(":")[:3])
    except (ValueError, TypeError):
        return 0
    now = datetime.now(tz=timezone.utc)
    return int(now.replace(hour=hour, minute=minute, second=second, microsecond=0).timestamp() * 1000)


class IssQuoteSource:
    """`get_order_book` через публичный ISS (без токена и без gRPC)."""

    def __init__(self, meta: InstrumentMetadata, depth: int = 5, timeout: int = 10) -> None:
        self._meta = meta
        self._depth = max(int(depth), 1)
        self._timeout = int(timeout)

    def _fetch_sync(self) -> Optional[OrderBookSnapshot]:
        request = urllib.request.Request(
            _url(self._meta, self._depth),
            headers={"User-Agent": "BotA-Scalper/3.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 — сеть/ISS может быть недоступен, поллинг повторит
            return None
        return _parse(self._meta, payload, self._depth)

    async def get_order_book(
        self, instrument_uid: Optional[str] = None, depth: Optional[int] = None
    ) -> Optional[OrderBookSnapshot]:
        return await asyncio.to_thread(self._fetch_sync)
