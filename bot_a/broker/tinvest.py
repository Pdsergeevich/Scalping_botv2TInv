"""Тонкий адаптер над SDK ``tinkoff-invest-api`` (gRPC v2).

Импорт SDK выполняется лениво: бэктест, оптимизация и юнит-тесты работают
без установленного пакета, а живые режимы выдают понятную ошибку.
"""

from __future__ import annotations

import inspect
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Sequence

from bot_a.models.domain import (
    InstrumentMetadata,
    MarketEvent,
    MarketKind,
    OrderBookLevel,
    OrderBookSnapshot,
    OrderStatus,
    Side,
    TradePrint,
)
from bot_a.utils.quotation import decimal_to_quotation, quotation_from_proto
from bot_a.utils.time_provider import TimeProvider

SDK_INSTALL_HINT = (
    "Python SDK T-Invest не установлен. Установите пакет 'tinkoff-invest-api' "
    "(нужен интерпретатор Python 3.10–3.13: у grpcio/protobuf нет сборок под 3.14)."
    " Живые режимы (preflight/record/paper/live) без SDK недоступны."
)

RUB_CURRENCIES = {"rub", "rur", "rub_"}


class SdkUnavailableError(RuntimeError):
    """SDK T-Invest недоступен в текущем окружении."""


def sdk_available() -> bool:
    try:
        import tinkoff.invest  # noqa: F401
    except Exception:
        return False
    return True


def require_sdk() -> Any:
    try:
        import tinkoff.invest as ti
    except Exception as exc:  # pragma: no cover - зависит от окружения
        raise SdkUnavailableError(SDK_INSTALL_HINT) from exc
    return ti


def _response_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


async def _call_supported(func: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
    """Вызывает метод SDK, отбрасывая аргументы, которых нет в его сигнатуре.

    Это защищает от расхождений между минорными версиями SDK.
    """
    try:
        signature = inspect.signature(func)
        supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return await func(**supported)
    except (TypeError, ValueError):
        return await func(**kwargs)


def _direction(side: Side) -> Any:
    ti = require_sdk()
    if side is Side.BUY:
        return _response_attr(ti.OrderDirection, "ORDER_DIRECTION_BUY")
    return _response_attr(ti.OrderDirection, "ORDER_DIRECTION_SELL")


def _interval(name: str) -> Any:
    ti = require_sdk()
    mapping = {
        "1min": "CANDLE_INTERVAL_1_MIN",
        "5min": "CANDLE_INTERVAL_5_MIN",
        "15min": "CANDLE_INTERVAL_15_MIN",
        "1hour": "CANDLE_INTERVAL_HOUR",
        "1day": "CANDLE_INTERVAL_DAY",
    }
    attr = mapping.get(name, "CANDLE_INTERVAL_1_MIN")
    return getattr(ti.CandleInterval, attr)


@dataclass(frozen=True, slots=True)
class AccountInfo:
    account_id: str
    name: str
    access_level: str
    account_type: str
    opened_date: str = ""


@dataclass(frozen=True, slots=True)
class BrokerOrderState:
    """Состояние заявки на стороне брокера."""

    order_id: str
    status: OrderStatus
    lots_requested: int
    lots_executed: int
    executed_price: Decimal
    commission_rub: Decimal = Decimal("0")
    message: str = ""


def create_client(token: str, app_name: str = "BotA-Scalper", timeout: int = 20) -> Any:
    """Создаёт асинхронный клиент SDK."""
    if not token:
        raise SdkUnavailableError("Не задан токен T-Invest (tInvest.token или переменная окружения).")
    ti = require_sdk()
    return ti.AsyncClient(token, app_name=app_name, timeout=timeout)


@asynccontextmanager
async def async_client(token: str, app_name: str = "BotA-Scalper", timeout: int = 20) -> AsyncIterator[Any]:
    """Контекстный менеджер клиента SDK с гарантированным закрытием канала."""
    client = create_client(token, app_name, timeout)
    try:
        yield client
    finally:
        closer = getattr(client, "close", None)
        closed = False
        if closer is not None:
            try:
                result = closer()
                if inspect.isawaitable(result):
                    await result
                closed = True
            except Exception:
                closed = False
        if not closed:
            aexit = getattr(client, "__aexit__", None)
            if aexit is not None:
                try:
                    await aexit(None, None, None)
                except Exception:
                    pass


# ---------------------------------------------------------------------- счёт и инструменты
async def get_accounts(client: Any) -> list[AccountInfo]:
    response = await client.users.get_accounts()
    accounts = _response_attr(response, "accounts", default=[]) or []
    result: list[AccountInfo] = []
    for account in accounts:
        result.append(
            AccountInfo(
                account_id=str(_response_attr(account, "id", default="")),
                name=str(_response_attr(account, "name", default="")),
                access_level=str(_response_attr(account, "access_level", default="")),
                account_type=str(_response_attr(account, "type", default="")),
                opened_date=str(_response_attr(account, "opened_date", default="")),
            )
        )
    return result


async def _futures_margin(client: Any, instrument_uid: str) -> tuple[Decimal, Decimal]:
    try:
        response = await client.instruments.get_futures_margin(instrument_id=instrument_uid)
    except Exception:
        return Decimal(0), Decimal(0)
    buy = quotation_from_proto(_response_attr(response, "initial_margin_on_buy", default=None))
    sell = quotation_from_proto(_response_attr(response, "initial_margin_on_sell", default=None))
    return buy, sell


async def resolve_instrument(client: Any, ticker: str, class_code: str, market: MarketKind) -> InstrumentMetadata:
    """Резолвит инструмент в метаданные с корректной экономикой пункта цены."""
    ti = require_sdk()
    id_type = _response_attr(ti.InstrumentIdType, "INSTRUMENT_ID_TYPE_TICKER")
    response = None
    if market is MarketKind.FUTURES:
        response = await _call_supported(
            client.instruments.future_by,
            id_type=id_type,
            class_code=class_code,
            id=ticker,
            instrument_id=ticker,
        )
    else:
        response = await _call_supported(
            client.instruments.share_by,
            id_type=id_type,
            class_code=class_code,
            id=ticker,
            instrument_id=ticker,
        )
    instrument = _response_attr(response, "instrument", default=None)
    if instrument is None:
        raise RuntimeError(f"Инструмент {ticker} ({class_code}) не найден")

    uid = str(_response_attr(instrument, "instrument_uid", default=""))
    lot = int(_response_attr(instrument, "lot", default=1) or 1)
    step = quotation_from_proto(_response_attr(instrument, "min_price_increment", default=None))
    step_amount = quotation_from_proto(_response_attr(instrument, "min_price_increment_amount", default=None))
    if step <= 0:
        step = Decimal("0.01")
    if market is MarketKind.FUTURES and step_amount <= 0:
        step_amount = step

    margin_buy = Decimal(0)
    margin_sell = Decimal(0)
    if market is MarketKind.FUTURES:
        margin_buy, margin_sell = await _futures_margin(client, uid)
        if margin_buy <= 0:
            margin_buy = quotation_from_proto(_response_attr(instrument, "initial_margin_on_buy", default=None))
        if margin_sell <= 0:
            margin_sell = quotation_from_proto(_response_attr(instrument, "initial_margin_on_sell", default=None))

    return InstrumentMetadata(
        instrument_uid=uid,
        ticker=ticker,
        class_code=class_code,
        market=market,
        figi=str(_response_attr(instrument, "figi", default="")),
        name=str(_response_attr(instrument, "name", default="")),
        lot_size=lot,
        min_price_increment=step,
        min_price_increment_amount=step_amount,
        initial_margin_buy=margin_buy,
        initial_margin_sell=margin_sell,
        currency=str(_response_attr(instrument, "currency", default="rub")),
    )


async def resolve_pair(
    client: Any,
    leg1: InstrumentMetadata,
    leg2: InstrumentMetadata,
) -> tuple[InstrumentMetadata, InstrumentMetadata]:
    """Резолвит обе ноги пары (нужно для бэктеста по API и живых режимов)."""
    first = await resolve_instrument(client, leg1.ticker, leg1.class_code, leg1.market)
    second = await resolve_instrument(client, leg2.ticker, leg2.class_code, leg2.market)
    return first, second


async def get_positions_money(client: Any, account_id: str) -> tuple[Decimal, Decimal]:
    """Возвращает ``(свободные_рубли, заблокированные_рубли)``."""
    response = await client.operations.get_positions(account_id=account_id)
    money = _response_attr(response, "money", default=[]) or []
    free = Decimal(0)
    blocked = Decimal(0)
    for item in money:
        currency = str(_response_attr(item, "currency", default="rub")).lower()
        if currency not in RUB_CURRENCIES:
            continue
        free += quotation_from_proto(item)
    blocked_items = _response_attr(response, "blocked", default=[]) or []
    for item in blocked_items:
        currency = str(_response_attr(item, "currency", default="rub")).lower()
        if currency not in RUB_CURRENCIES:
            continue
        blocked += quotation_from_proto(item)
    return free, blocked


async def is_trading_available(client: Any, instrument_uid: str) -> tuple[bool, str]:
    try:
        status = await client.market_data.get_trading_status(instrument_id=instrument_uid)
    except Exception as exc:
        return False, f"статус торговли недоступен: {exc}"
    available = bool(_response_attr(status, "api_trade_available_flag", default=False))
    limit_up = quotation_from_proto(_response_attr(status, "limit_up", default=None))
    limit_down = quotation_from_proto(_response_attr(status, "limit_down", default=None))
    detail = f"api_trade_available={available}, limit_down={limit_down}, limit_up={limit_up}"
    return available, detail


async def get_order_book(client: Any, instrument_uid: str, depth: int = 50) -> Optional[OrderBookSnapshot]:
    try:
        book = await client.market_data.get_order_book(instrument_id=instrument_uid, depth=depth)
    except Exception:
        return None
    return _parse_order_book(book, instrument_uid, 0)


def _parse_order_book(book: Any, instrument_uid: str, fallback_time_ms: int) -> OrderBookSnapshot:
    bids = tuple(
        OrderBookLevel(price=quotation_from_proto(level.price), quantity=int(level.quantity))
        for level in (_response_attr(book, "bids", default=[]) or [])
    )
    asks = tuple(
        OrderBookLevel(price=quotation_from_proto(level.price), quantity=int(level.quantity))
        for level in (_response_attr(book, "asks", default=[]) or [])
    )
    time_ms = fallback_time_ms
    timestamp = _response_attr(book, "time", default=None)
    if isinstance(timestamp, datetime):
        time_ms = int(timestamp.timestamp() * 1000)
    last_price = _response_attr(book, "last_price", default=None)
    return OrderBookSnapshot(
        instrument_uid=str(_response_attr(book, "instrument_uid", default=instrument_uid)),
        time_ms=time_ms,
        bids=bids,
        asks=asks,
        last_price=quotation_from_proto(last_price) if last_price is not None else None,
    )


# ---------------------------------------------------------------------- свечи
async def get_candles(
    client: Any,
    instrument_uid: str,
    date_from: datetime,
    date_to: datetime,
    interval: str = "1min",
    max_chunk_days: int = 1,
) -> list[dict]:
    """Постранично загружает свечи (по умолчанию — минутные, окно 1 день)."""
    interval_enum = _interval(interval)
    result: list[dict] = []
    cursor = date_from.astimezone(timezone.utc)
    end = date_to.astimezone(timezone.utc)
    if cursor.tzinfo is None:
        cursor = cursor.replace(tzinfo=timezone.utc)
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=max_chunk_days), end)
        response = await _call_supported(
            client.market_data.get_candles,
            instrument_id=instrument_uid,
            instrument_uid=instrument_uid,
            from_=cursor,
            to=chunk_end,
            interval=interval_enum,
        )
        candles = _response_attr(response, "candles", default=[]) or []
        for candle in candles:
            moment = _response_attr(candle, "time", default=None)
            moment_ms = int(moment.timestamp() * 1000) if isinstance(moment, datetime) else 0
            result.append(
                {
                    "time": moment_ms,
                    "open": str(quotation_from_proto(_response_attr(candle, "open", default=None))),
                    "high": str(quotation_from_proto(_response_attr(candle, "high", default=None))),
                    "low": str(quotation_from_proto(_response_attr(candle, "low", default=None))),
                    "close": str(quotation_from_proto(_response_attr(candle, "close", default=None))),
                    "volume": int(_response_attr(candle, "volume", default=0) or 0),
                }
            )
        cursor = chunk_end + timedelta(milliseconds=1)
    return result


# ---------------------------------------------------------------------- потоковый маркет-дата
class TinkoffMarketDataStream:
    """Подписка на стаканы (100 мс) и ленту сделок через gRPC-стрим."""

    def __init__(
        self,
        client: Any,
        legs: Sequence[InstrumentMetadata],
        depth: int = 5,
        clock: Optional[TimeProvider] = None,
    ) -> None:
        self._client = client
        self._legs = list(legs)
        self._depth = int(depth)
        self._clock = clock
        self._uid_to_leg = {meta.instrument_uid: index + 1 for index, meta in enumerate(self._legs)}
        self._manager_cm: Any = None
        self._manager: Any = None
        self._order_book_iterator: Any = None
        self._trades_iterator: Any = None

    def _fallback_time_ms(self) -> int:
        return self._clock.now_ms() if self._clock is not None else 0

    async def __aenter__(self) -> "TinkoffMarketDataStream":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def open(self) -> None:
        factory = getattr(self._client, "create_market_data_stream", None)
        if factory is not None:
            self._manager_cm = factory()
            self._manager = await self._manager_cm.__aenter__()
            for meta in self._legs:
                self._manager.order_book.subscribe(instrument_id=meta.instrument_uid, depth=self._depth)
                subscribe_trades = getattr(self._manager, "trades", None)
                if subscribe_trades is not None:
                    subscribe_trades.subscribe(instrument_id=meta.instrument_uid)
            return
        # Совместимость со старыми версиями SDK, где стримы вызываются напрямую.
        ti = require_sdk()
        request = ti.SubscribeOrderBookRequest(
            subscription=ti.OrderBookSubscription(
                instrument_id=self._legs[0].instrument_uid,
                depth=self._depth,
            )
        )
        self._order_book_iterator = self._client.market_data.order_book_stream(subscribe_order_book_request=request)
        self._trades_iterator = self._client.market_data.trades_stream(
            instruments=[ti.TradesSubscription(instrument_id=meta.instrument_uid) for meta in self._legs]
        )

    async def close(self) -> None:
        if self._manager is not None:
            try:
                for meta in self._legs:
                    for subscriber in ("order_book", "trades"):
                        part = getattr(self._manager, subscriber, None)
                        if part is not None and hasattr(part, "unsubscribe"):
                            part.unsubscribe(instrument_id=meta.instrument_uid)
            finally:
                try:
                    await self._manager_cm.__aexit__(None, None, None)
                except Exception:
                    pass
                self._manager_cm = None
                self._manager = None
        if self._order_book_iterator is not None and hasattr(self._order_book_iterator, "cancel"):
            self._order_book_iterator.cancel()
            self._order_book_iterator = None
        if self._trades_iterator is not None and hasattr(self._trades_iterator, "cancel"):
            self._trades_iterator.cancel()
            self._trades_iterator = None

    async def events(self) -> AsyncIterator[MarketEvent]:
        """Бесконечный поток нормализованных событий рынка."""
        if self._manager is not None:
            async for response in self._manager:
                event = self._parse_response(response)
                if event is not None:
                    yield event
            return
        if self._order_book_iterator is not None:
            async for response in self._order_book_iterator:
                book = _response_attr(response, "orderbook", "order_book", default=None)
                if book is not None:
                    yield self._parse_book(book)
        if self._trades_iterator is not None:
            async for response in self._trades_iterator:
                trade = _response_attr(response, "trade", "trades", default=None)
                if isinstance(trade, list):
                    for item in trade:
                        event = self._parse_trade(item)
                        if event is not None:
                            yield event
                elif trade is not None:
                    event = self._parse_trade(trade)
                    if event is not None:
                        yield event

    def _parse_response(self, response: Any) -> Optional[MarketEvent]:
        book = _response_attr(response, "orderbook", "order_book", default=None)
        if book is not None:
            return self._parse_book(book)
        trade = _response_attr(response, "trade", default=None)
        if trade is not None:
            return self._parse_trade(trade)
        return None

    def _parse_book(self, book: Any) -> Optional[MarketEvent]:
        uid = str(_response_attr(book, "instrument_uid", default=""))
        leg = self._uid_to_leg.get(uid)
        if leg is None:
            return None
        snapshot = _parse_order_book(book, uid, self._fallback_time_ms())
        return MarketEvent.of_book(leg, snapshot)

    def _parse_trade(self, trade: Any) -> Optional[MarketEvent]:
        uid = str(_response_attr(trade, "instrument_uid", default=""))
        leg = self._uid_to_leg.get(uid)
        if leg is None:
            return None
        direction = _response_attr(trade, "direction", default=None)
        side: Optional[Side] = None
        direction_text = str(direction)
        if "BUY" in direction_text:
            side = Side.BUY
        elif "SELL" in direction_text:
            side = Side.SELL
        moment = _response_attr(trade, "time", default=None)
        time_ms = int(moment.timestamp() * 1000) if isinstance(moment, datetime) else self._fallback_time_ms()
        print_ = TradePrint(
            instrument_uid=uid,
            time_ms=time_ms,
            price=quotation_from_proto(_response_attr(trade, "price", default=None)),
            quantity=int(_response_attr(trade, "quantity", default=0) or 0),
            side=side,
        )
        return MarketEvent.of_trade(leg, print_)


# ---------------------------------------------------------------------- заявки
class TinkoffOrderClient:
    """Обёртка над ``OrdersService`` для живого шлюза."""

    def __init__(self, client: Any, account_id: str, legs: Sequence[InstrumentMetadata]) -> None:
        self._client = client
        self._account_id = account_id
        self._legs = list(legs)

    def meta(self, leg: int) -> InstrumentMetadata:
        return self._legs[leg - 1]

    async def place(
        self,
        leg: int,
        side: Side,
        order_type: str,
        quantity: int,
        price: Optional[Decimal],
        order_id: Optional[str] = None,
    ) -> tuple[bool, str, str]:
        ti = require_sdk()
        meta = self.meta(leg)
        is_market = order_type == "market"
        sdk_type = (
            _response_attr(ti.OrderType, "ORDER_TYPE_MARKET")
            if is_market
            else _response_attr(ti.OrderType, "ORDER_TYPE_LIMIT")
        )
        kwargs: dict[str, Any] = {
            "instrument_id": meta.instrument_uid,
            "quantity": int(quantity),
            "direction": _direction(side),
            "account_id": self._account_id,
            "order_type": sdk_type,
            "order_id": order_id or str(uuid.uuid4()),
        }
        # Брокер требует Quotation всегда: для рыночной заявки передаём нулевую цену.
        units, nano = decimal_to_quotation(price if price is not None else Decimal(0))
        kwargs["price"] = ti.Quotation(units=units, nano=nano)
        if order_type == "limit_ioc":
            kwargs["time_in_force"] = _response_attr(ti.TimeInForce, "TIME_IN_FORCE_FILL_AND_KILL")
        response = await _call_supported(self._client.orders.post_order, **kwargs)
        response_order_id = str(_response_attr(response, "order_id", default=kwargs["order_id"]))
        return True, response_order_id, ""

    async def replace(self, order_id: str, quantity: int, price: Decimal) -> tuple[bool, str]:
        ti = require_sdk()
        units, nano = decimal_to_quotation(price)
        try:
            await _call_supported(
                self._client.orders.replace_order,
                account_id=self._account_id,
                order_id=order_id,
                quantity=int(quantity),
                price=ti.Quotation(units=units, nano=nano),
                idempotency_key=str(uuid.uuid4()),
            )
            return True, ""
        except Exception as exc:
            return False, str(exc)

    async def cancel(self, order_id: str) -> tuple[bool, str]:
        try:
            await _call_supported(
                self._client.orders.cancel_order,
                account_id=self._account_id,
                order_id=order_id,
            )
            return True, ""
        except Exception as exc:
            return False, str(exc)

    async def get_state(self, order_id: str) -> Optional[BrokerOrderState]:
        try:
            state = await _call_supported(
                self._client.orders.get_order_state,
                account_id=self._account_id,
                order_id=order_id,
            )
        except Exception:
            return None
        status = str(_response_attr(state, "execution_report_status", default=""))
        mapped = OrderStatus.NEW
        if "FILL" in status and "PARTIAL" not in status:
            mapped = OrderStatus.FILLED
        elif "PARTIAL" in status:
            mapped = OrderStatus.PARTIALLY_FILLED
        elif "CANCEL" in status:
            mapped = OrderStatus.CANCELLED
        elif "REJECT" in status:
            mapped = OrderStatus.REJECTED
        executed_price = quotation_from_proto(
            _response_attr(state, "executed_order_price", "average_position_price", default=None)
        )
        commission = quotation_from_proto(_response_attr(state, "service_commission", default=None))
        return BrokerOrderState(
            order_id=str(_response_attr(state, "order_id", default=order_id)),
            status=mapped,
            lots_requested=int(_response_attr(state, "lots_requested", default=0) or 0),
            lots_executed=int(_response_attr(state, "lots_executed", default=0) or 0),
            executed_price=executed_price,
            commission_rub=commission,
        )

    async def free_cash_rub(self) -> Decimal:
        free, _ = await get_positions_money(self._client, self._account_id)
        return free
