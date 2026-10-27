"""Нативный REST/JSON клиент T-Invest (без gRPC SDK).

Работает на любом Python (включая 3.14), где нет колёс grpcio/protobuf.
Использует официальный REST-шлюз ``https://invest-public-api.tinkoff.ru/rest``
и protobuf-JSON-представление сообщений.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence

from bot_a.models.domain import (
    InstrumentMetadata,
    MarketKind,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
    TradePrint,
)
from bot_a.utils.quotation import dec

REST_BASE = "https://invest-public-api.tinkoff.ru/rest"
PROTO_PREFIX = "/tinkoff.public.invest.api.contract.v1."

APP_NAME = "BotA-Scalper-REST"


class TInvestRestError(RuntimeError):
    """Ошибка REST-шлюза T-Invest."""

    def __init__(self, message: str, status: int = 0, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def quotation_to_dict(value: Decimal) -> Dict[str, Any]:
    """``Decimal`` -> protobuf-JSON Quotation ``{"units": "...", "nano": N}``."""
    amount = abs(dec(value))
    units = int(amount)
    nano = int((amount - Decimal(units)) * Decimal(1_000_000_000))
    if dec(value) < 0:
        return {"units": str(-units), "nano": -nano}
    return {"units": str(units), "nano": nano}


def quotation_from_dict(payload: Optional[Dict[str, Any]]) -> Decimal:
    if not payload:
        return Decimal(0)
    units = int(payload.get("units", 0) or 0)
    nano = int(payload.get("nano", 0) or 0)
    return Decimal(units) + Decimal(nano) / Decimal(1_000_000_000)


def datetime_to_rfc3339(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rfc3339_to_ms(value: Optional[str]) -> int:
    if not value:
        return 0
    text = value.replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return 0
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp() * 1000)


@dataclass
class RestOrderState:
    order_id: str
    status: str
    lots_requested: int
    lots_executed: int
    executed_price: Decimal


class TInvestRestClient:
    """Минимальный клиент нужных методов T-Invest через REST/JSON."""

    def __init__(
        self,
        token: str,
        app_name: str = APP_NAME,
        base_url: str = REST_BASE,
        timeout: int = 20,
        transport: Optional[Callable[[str, Dict[str, Any], Dict[str, str]], Dict[str, Any]]] = None,
    ) -> None:
        if not token:
            raise TInvestRestError("Не задан токен T-Invest для REST-клиента")
        token = token.replace("\ufeff", "").strip().strip('"').strip("'").strip()
        if not token:
            raise TInvestRestError("Токен T-Invest пуст после очистки (BOM/кавычки)")
        try:
            token.encode("latin-1")
        except UnicodeEncodeError as exc:
            raise TInvestRestError(
                "Токен содержит недопустимые символы (BOM или юникод). "
                "Запишите файл без BOM: Set-Content -Encoding ascii ... или уберите BOM"
            ) from exc
        self._token = token
        self._app_name = app_name
        self._base_url = base_url.rstrip("/")
        self._timeout = int(timeout)
        self._transport = transport

    # ------------------------------------------------------------------ транспорт
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-app-name": self._app_name,
        }

    def _post_sync(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        headers = self._headers()
        if self._transport is not None:
            try:
                return self._transport(url, payload, headers)
            except (TimeoutError, socket.timeout) as exc:
                raise TInvestRestError(f"Таймаут ответа за {self._timeout} с") from exc
            except (ConnectionError, http.client.HTTPException) as exc:
                raise TInvestRestError(f"Сбой транспорта: {type(exc).__name__}: {exc}") from exc
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise TInvestRestError(f"HTTP {exc.code}: {detail}", exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise TInvestRestError(f"Сеть недоступна: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            # Долгий ответ не должен убивать сессию: поллинг продолжает работу.
            raise TInvestRestError(f"Таймаут ответа за {self._timeout} с") from exc
        except (ConnectionError, http.client.HTTPException, json.JSONDecodeError) as exc:
            raise TInvestRestError(f"Сбой транспорта: {type(exc).__name__}: {exc}") from exc

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await asyncio.to_thread(self._post_sync, path, payload)

    @staticmethod
    def _path(service: str, method: str) -> str:
        return f"{PROTO_PREFIX}{service}/{method}"

    # ------------------------------------------------------------------ справочники
    async def get_accounts(self) -> List[Dict[str, Any]]:
        response = await self._post(self._path("UsersService", "GetAccounts"), {})
        return list(response.get("accounts") or [])

    async def future_by(self, ticker: str, class_code: str = "SPBFUT") -> Dict[str, Any]:
        response = await self._post(
            self._path("InstrumentsService", "FutureBy"),
            {"idType": "INSTRUMENT_ID_TYPE_TICKER", "classCode": class_code, "id": ticker},
        )
        return dict(response.get("instrument") or {})

    async def share_by(self, ticker: str, class_code: str = "TQBR") -> Dict[str, Any]:
        response = await self._post(
            self._path("InstrumentsService", "ShareBy"),
            {"idType": "INSTRUMENT_ID_TYPE_TICKER", "classCode": class_code, "id": ticker},
        )
        return dict(response.get("instrument") or {})

    async def get_futures_margin(self, instrument_uid: str) -> Dict[str, Any]:
        return await self._post(
            self._path("InstrumentsService", "GetFuturesMargin"),
            {"instrumentId": instrument_uid},
        )

    async def resolve_instrument(self, ticker: str, class_code: str, market: MarketKind) -> InstrumentMetadata:
        if market is MarketKind.FUTURES:
            raw = await self.future_by(ticker, class_code)
            margin = await self.get_futures_margin(str(raw.get("uid") or raw.get("instrumentUid") or ""))
            buy = quotation_from_dict(margin.get("initialMarginOnBuy"))
            sell = quotation_from_dict(margin.get("initialMarginOnSell"))
            step = quotation_from_dict(raw.get("minPriceIncrement")) or quotation_from_dict(
                margin.get("minPriceIncrement")
            )
            step_amount = quotation_from_dict(raw.get("minPriceIncrementAmount")) or quotation_from_dict(
                margin.get("minPriceIncrementAmount")
            )
        else:
            raw = await self.share_by(ticker, class_code)
            buy = sell = Decimal(0)
            step = quotation_from_dict(raw.get("minPriceIncrement")) or Decimal("0.01")
            step_amount = Decimal(0)
        lot = int(raw.get("lot", 1) or 1)
        return InstrumentMetadata(
            instrument_uid=str(raw.get("uid") or raw.get("instrumentUid") or ""),
            ticker=ticker,
            class_code=class_code,
            market=market,
            figi=str(raw.get("figi") or ""),
            name=str(raw.get("name") or ticker),
            lot_size=lot,
            min_price_increment=step if step > 0 else Decimal("0.01"),
            min_price_increment_amount=step_amount,
            initial_margin_buy=buy,
            initial_margin_sell=sell,
            currency=str(raw.get("currency") or "rub"),
        )

    # ------------------------------------------------------------------ рыночные данные
    async def get_candles(
        self,
        instrument_uid: str,
        date_from: datetime,
        date_to: datetime,
        interval: str = "CANDLE_INTERVAL_1_MIN",
    ) -> List[Dict[str, Any]]:
        response = await self._post(
            self._path("MarketDataService", "GetCandles"),
            {
                "instrumentId": instrument_uid,
                "from": datetime_to_rfc3339(date_from),
                "to": datetime_to_rfc3339(date_to),
                "interval": interval,
            },
        )
        return list(response.get("candles") or [])

    async def get_order_book(self, instrument_uid: str, depth: int = 20) -> Optional[OrderBookSnapshot]:
        response = await self._post(
            self._path("MarketDataService", "GetOrderBook"),
            {"instrumentId": instrument_uid, "depth": int(depth)},
        )
        if not response:
            return None
        bids = tuple(
            OrderBookLevel(
                price=quotation_from_dict(level.get("price")),
                quantity=int(level.get("quantity", 0) or 0),
            )
            for level in (response.get("bids") or [])
        )
        asks = tuple(
            OrderBookLevel(
                price=quotation_from_dict(level.get("price")),
                quantity=int(level.get("quantity", 0) or 0),
            )
            for level in (response.get("asks") or [])
        )
        last_price = response.get("lastPrice")
        return OrderBookSnapshot(
            instrument_uid=str(response.get("instrumentUid") or instrument_uid),
            time_ms=rfc3339_to_ms(response.get("time")) or int(datetime.now(tz=timezone.utc).timestamp() * 1000),
            bids=bids,
            asks=asks,
            last_price=quotation_from_dict(last_price) if last_price else None,
        )

    async def get_last_trades(
        self,
        instrument_uid: str,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[TradePrint]:
        payload: Dict[str, Any] = {"instrumentId": instrument_uid}
        # T-Invest REST требует обе границы: если задан только `from`, сервер
        # подставляет `to` = epoch и отвечает «to can't be less than from».
        if date_from is not None:
            payload["from"] = datetime_to_rfc3339(date_from)
            payload["to"] = datetime_to_rfc3339(
                date_to if date_to is not None else datetime.now(tz=timezone.utc)
            )
        elif date_to is not None:
            payload["to"] = datetime_to_rfc3339(date_to)
        response = await self._post(self._path("MarketDataService", "GetLastTrades"), payload)
        trades: List[TradePrint] = []
        for item in response.get("trades") or []:
            direction = str(item.get("direction") or "")
            side = Side.BUY if "BUY" in direction else (Side.SELL if "SELL" in direction else None)
            trades.append(
                TradePrint(
                    instrument_uid=instrument_uid,
                    time_ms=rfc3339_to_ms(item.get("time")),
                    price=quotation_from_dict(item.get("price")),
                    quantity=int(item.get("quantity", 0) or 0),
                    side=side,
                )
            )
        return trades

    async def get_trading_status(self, instrument_uid: str) -> Dict[str, Any]:
        return await self._post(
            self._path("MarketDataService", "GetTradingStatus"),
            {"instrumentId": instrument_uid},
        )

    # ------------------------------------------------------------------ заявки
    async def post_order(
        self,
        instrument_uid: str,
        quantity: int,
        direction: Side,
        account_id: str,
        order_id: str,
        price: Optional[Decimal] = None,
        order_type: str = "ORDER_TYPE_LIMIT",
        time_in_force: str = "TIME_IN_FORCE_DAY",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "instrumentId": instrument_uid,
            "quantity": str(int(quantity)),
            "direction": "ORDER_DIRECTION_BUY" if direction is Side.BUY else "ORDER_DIRECTION_SELL",
            "accountId": account_id,
            "orderType": order_type,
            "orderId": order_id,
            "timeInForce": time_in_force,
            "price": quotation_to_dict(price or Decimal(0)),
        }
        return await self._post(self._path("OrdersService", "PostOrder"), payload)

    async def cancel_order(self, account_id: str, order_id: str) -> Dict[str, Any]:
        return await self._post(
            self._path("OrdersService", "CancelOrder"),
            {"accountId": account_id, "orderId": order_id},
        )

    async def replace_order(
        self,
        account_id: str,
        order_id: str,
        quantity: int,
        price: Decimal,
        idempotency_key: str,
    ) -> Dict[str, Any]:
        return await self._post(
            self._path("OrdersService", "ReplaceOrder"),
            {
                "accountId": account_id,
                "orderId": order_id,
                "quantity": str(int(quantity)),
                "price": quotation_to_dict(price),
                "priceType": "PRICE_TYPE_POINT",
                "idempotencyKey": idempotency_key,
            },
        )

    async def get_order_state(self, account_id: str, order_id: str) -> Optional[RestOrderState]:
        try:
            payload = await self._post(
                self._path("OrdersService", "GetOrderState"),
                {"accountId": account_id, "orderId": order_id},
            )
        except TInvestRestError:
            return None
        return RestOrderState(
            order_id=str(payload.get("orderId") or order_id),
            status=str(payload.get("executionReportStatus") or ""),
            lots_requested=int(payload.get("lotsRequested", 0) or 0),
            lots_executed=int(payload.get("lotsExecuted", 0) or 0),
            executed_price=quotation_from_dict(payload.get("executedOrderPrice")),
        )

    # ------------------------------------------------------------------ счёт
    async def get_positions(self, account_id: str) -> Dict[str, Any]:
        return await self._post(
            self._path("OperationsService", "GetPositions"), {"accountId": account_id}
        )

    async def free_cash_rub(self, account_id: str) -> Decimal:
        payload = await self.get_positions(account_id)
        total = Decimal(0)
        for item in payload.get("money") or []:
            if str(item.get("currency", "")).lower() not in ("rub", "rur"):
                continue
            total += quotation_from_dict(item)
        return total


class RestRecordingService:
    """Запись L2 через REST-поллинг ``GetOrderBook`` (+ ``GetLastTrades``)."""

    def __init__(
        self,
        config: Any,
        preset: Any,
        client: TInvestRestClient,
        interval_ms: int = 150,
        depth: int = 20,
        trades_every_seconds: float = 1.0,
        logger: Any = None,
    ) -> None:
        import logging

        from bot_a.backtest.recorder import MarketDataRecorder  # локальный импорт против цикла

        self._config = config
        self._preset = preset
        self._client = client
        self._interval = max(int(interval_ms), 50)
        self._depth = max(int(depth), 1)
        self._trades_every = max(float(trades_every_seconds), 0.2)
        self._log = logger or logging.getLogger("bot_a.recorder.rest")
        self._recorder_factory = MarketDataRecorder

    async def run(self, duration_seconds: Optional[float] = None, stop_event: Any = None) -> Dict[str, Any]:
        import time

        from bot_a.models.domain import MarketEvent

        recorder = self._recorder_factory(self._config, self._preset, self._log)
        await recorder.start()
        legs = [(1, self._preset.leg1), (2, self._preset.leg2)]
        deadline = (time.monotonic() + duration_seconds) if duration_seconds else None
        last_trades = 0.0
        seen: set[tuple] = set()
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if stop_event is not None and stop_event.is_set():
                    break
                for leg, meta in legs:
                    try:
                        book = await self._client.get_order_book(meta.instrument_uid, self._depth)
                    except TInvestRestError as exc:
                        self._log.warning("GetOrderBook %s: %s", meta.ticker, exc)
                        continue
                    if book is not None:
                        recorder.write(MarketEvent.of_book(leg, book))
                now = time.monotonic()
                if now - last_trades >= self._trades_every:
                    last_trades = now
                    # Окно 60 с: REST отдаёт принты с задержкой, 5-секундное окно пустое.
                    since = datetime.now(tz=timezone.utc) - timedelta(seconds=60)
                    for leg, meta in legs:
                        try:
                            trades = await self._client.get_last_trades(meta.instrument_uid, since)
                        except TInvestRestError as exc:
                            self._log.warning("GetLastTrades %s: %s", meta.ticker, exc)
                            continue
                        for trade in trades:
                            key = (meta.instrument_uid, trade.time_ms, str(trade.price), trade.quantity)
                            if key in seen:
                                continue
                            seen.add(key)
                            recorder.write(MarketEvent.of_trade(leg, trade))
                    if len(seen) > 50000:
                        seen.clear()
                await asyncio.sleep(self._interval / 1000.0)
        finally:
            await recorder.close()
        return recorder.stats.to_dict()
