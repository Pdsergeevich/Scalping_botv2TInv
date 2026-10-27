"""Тесты нативного REST-клиента T-Invest (без сети, через подмену транспорта)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Tuple

from bot_a.broker.tinkoff_rest_client import (
    TInvestRestClient,
    quotation_from_dict,
    quotation_to_dict,
    rfc3339_to_ms,
)
from bot_a.models.domain import MarketKind, Side


def test_quotation_json_round_trip():
    assert quotation_to_dict(Decimal("12.5")) == {"units": "12", "nano": 500_000_000}
    assert quotation_to_dict(Decimal("-0.001")) == {"units": "0", "nano": -1_000_000}
    assert quotation_from_dict({"units": "12", "nano": 500_000_000}) == Decimal("12.5")
    assert quotation_from_dict({"units": "-2", "nano": -250_000_000}) == Decimal("-2.25")
    assert quotation_from_dict(None) == Decimal(0)


def test_rfc3339_parsing():
    assert rfc3339_to_ms("1970-01-01T00:00:01Z") == 1000
    assert rfc3339_to_ms("") == 0
    assert rfc3339_to_ms("not-a-date") == 0


class FakeTransport:
    """Имитация REST-шлюза: проверяет путь и тело запроса."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def __call__(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        self.calls.append((url, payload))
        assert headers["Authorization"].startswith("Bearer ")
        if url.endswith("UsersService/GetAccounts"):
            return {"accounts": [{"id": "acc-1", "accessLevel": "ACCOUNT_ACCESS_LEVEL_FULL_ACCESS"}]}
        if url.endswith("InstrumentsService/FutureBy"):
            return {
                "instrument": {
                    "uid": "uid-CRZ6",
                    "figi": "FUT.CRZ6",
                    "name": "CNY-12.26",
                    "lot": 1,
                    "currency": "rub",
                    "minPriceIncrement": {"units": "0", "nano": 1_000_000},
                    "minPriceIncrementAmount": {"units": "1", "nano": 0},
                }
            }
        if url.endswith("InstrumentsService/GetFuturesMargin"):
            return {
                "initialMarginOnBuy": {"units": "1174", "nano": 940_000_000},
                "initialMarginOnSell": {"units": "1174", "nano": 940_000_000},
            }
        if url.endswith("MarketDataService/GetOrderBook"):
            return {
                "instrumentUid": "uid-CRZ6",
                "time": "2026-10-20T10:00:00Z",
                "bids": [{"price": {"units": "12", "nano": 500_000_000}, "quantity": "15"}],
                "asks": [{"price": {"units": "12", "nano": 510_000_000}, "quantity": "20"}],
                "lastPrice": {"units": "12", "nano": 505_000_000},
            }
        if url.endswith("MarketDataService/GetLastTrades"):
            return {
                "trades": [
                    {
                        "price": {"units": "12", "nano": 505_000_000},
                        "quantity": "3",
                        "time": "2026-10-20T10:00:01Z",
                        "direction": "ORDER_DIRECTION_BUY",
                    }
                ]
            }
        if url.endswith("OrdersService/PostOrder"):
            return {"orderId": payload["orderId"]}
        if url.endswith("OrdersService/CancelOrder"):
            return {}
        if url.endswith("OperationsService/GetPositions"):
            return {"money": [{"currency": "rub", "units": "30000", "nano": 0}]}
        return {}


def _client() -> tuple[TInvestRestClient, FakeTransport]:
    transport = FakeTransport()
    client = TInvestRestClient("test-token", transport=transport)
    return client, transport


def test_resolve_futures_metadata_and_margin():
    async def scenario():
        client, transport = _client()
        meta = await client.resolve_instrument("CRZ6", "SPBFUT", MarketKind.FUTURES)
        assert meta.instrument_uid == "uid-CRZ6"
        assert meta.min_price_increment == Decimal("0.001")
        assert meta.min_price_increment_amount == Decimal("1")
        assert meta.initial_margin_buy == Decimal("1174.94")
        assert meta.lot_size == 1
        paths = [url.rsplit("/", 1)[-1] for url, _ in transport.calls]
        assert paths == ["FutureBy", "GetFuturesMargin"]
        assert transport.calls[0][1]["idType"] == "INSTRUMENT_ID_TYPE_TICKER"
        assert transport.calls[0][1]["classCode"] == "SPBFUT"
        assert transport.calls[1][1]["instrumentId"] == "uid-CRZ6"

    asyncio.run(scenario())


def test_order_book_and_trades_parsing():
    async def scenario():
        client, _ = _client()
        book = await client.get_order_book("uid-CRZ6", depth=20)
        assert book is not None
        assert book.best_bid == Decimal("12.5")
        assert book.best_ask == Decimal("12.51")
        assert book.bids[0].quantity == 15
        assert book.time_ms == rfc3339_to_ms("2026-10-20T10:00:00Z")
        trades = await client.get_last_trades("uid-CRZ6", datetime(2026, 10, 20, 9, 59, 0, tzinfo=timezone.utc))
        assert len(trades) == 1
        assert trades[0].price == Decimal("12.505")
        assert trades[0].side is Side.BUY
        assert trades[0].quantity == 3

    asyncio.run(scenario())


def test_post_and_cancel_order_payloads():
    async def scenario():
        client, transport = _client()
        response = await client.post_order(
            "uid-CRZ6", 2, Side.SELL, "acc-1", "order-1", Decimal("12.505"), time_in_force="TIME_IN_FORCE_FILL_AND_KILL"
        )
        assert response["orderId"] == "order-1"
        payload = transport.calls[-1][1]
        assert payload["direction"] == "ORDER_DIRECTION_SELL"
        assert payload["quantity"] == "2"
        assert payload["price"] == {"units": "12", "nano": 505_000_000}
        assert payload["timeInForce"] == "TIME_IN_FORCE_FILL_AND_KILL"
        await client.cancel_order("acc-1", "order-1")
        assert transport.calls[-1][1] == {"accountId": "acc-1", "orderId": "order-1"}

    asyncio.run(scenario())


def test_free_cash_and_accounts():
    async def scenario():
        client, _ = _client()
        assert await client.free_cash_rub("acc-1") == Decimal("30000")
        accounts = await client.get_accounts()
        assert accounts[0]["id"] == "acc-1"

    asyncio.run(scenario())
