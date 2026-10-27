"""Абстракция шлюза исполнения.

Движок (``PairScalpingStateMachine``) ничего не знает о том, откуда пришли
данные и куда уходят заявки: он работает только с этим интерфейсом.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional, Sequence

from bot_a.economics.fees import TinkoffFeeCalculator
from bot_a.models.domain import (
    InstrumentMetadata,
    Liquidity,
    OrderBookSnapshot,
    OrderStatus,
    OrderType,
    Side,
    TradePrint,
)
from bot_a.utils.channel import Channel
from bot_a.utils.time_provider import TimeProvider

ZERO = Decimal("0")


class ExecutionEventKind(str, Enum):
    ACK = "ack"
    FILL = "fill"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """Запрос на выставление заявки."""

    leg: int
    side: Side
    order_type: OrderType
    quantity: int
    price: Optional[Decimal] = None
    client_tag: str = ""


@dataclass(frozen=True, slots=True)
class OrderAck:
    """Ответ на запрос выставления/замены заявки."""

    order_id: str
    leg: int
    side: Side
    order_type: OrderType
    price: Optional[Decimal]
    quantity: int
    accepted: bool
    time_ms: int = 0
    message: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """Событие исполнения, публикуемое шлюзом в канал движка."""

    kind: ExecutionEventKind
    order_id: str
    leg: int
    side: Side
    price: Decimal
    quantity: int
    remaining: int
    time_ms: int
    liquidity: Liquidity = Liquidity.MAKER
    commission_rub: Decimal = ZERO
    status: OrderStatus = OrderStatus.NEW
    message: str = ""

    @property
    def is_fill(self) -> bool:
        return self.kind is ExecutionEventKind.FILL

    @property
    def is_partial(self) -> bool:
        return self.kind is ExecutionEventKind.FILL and self.remaining > 0


@dataclass(slots=True)
class OrderState:
    """Локальное состояние заявки, синхронизированное со шлюзом."""

    order_id: str
    leg: int
    side: Side
    order_type: OrderType
    price: Optional[Decimal]
    quantity: int
    filled_quantity: int = 0
    status: OrderStatus = OrderStatus.NEW
    created_ms: int = 0
    updated_ms: int = 0
    client_tag: str = ""

    @property
    def remaining(self) -> int:
        return max(self.quantity - self.filled_quantity, 0)

    @property
    def is_active(self) -> bool:
        return not self.status.is_terminal

    def apply_fill(self, quantity: int, price: Decimal, time_ms: int) -> None:
        self.filled_quantity = min(self.quantity, self.filled_quantity + max(quantity, 0))
        if price > 0:
            self.price = price
        self.updated_ms = time_ms
        self.status = OrderStatus.FILLED if self.remaining <= 0 else OrderStatus.PARTIALLY_FILLED


class OrderExecutionGateway(abc.ABC):
    """Базовый шлюз: живой gRPC, бумажный симулятор или симулятор бэктеста."""

    def __init__(
        self,
        metadata: Sequence[InstrumentMetadata],
        clock: TimeProvider,
        fee_calculator: TinkoffFeeCalculator,
        event_sink: Optional[Channel[ExecutionEvent]] = None,
    ) -> None:
        if len(metadata) != 2:
            raise ValueError("Ожидаются метаданные двух ног пары")
        self._metadata = list(metadata)
        self._clock = clock
        self._fees = fee_calculator
        self.events: Channel[ExecutionEvent] = event_sink or Channel[ExecutionEvent]()
        self._orders: dict[str, OrderState] = {}
        self._seq = 0

    # ------------------------------------------------------------------ доступ
    @property
    def metadata(self) -> list[InstrumentMetadata]:
        return self._metadata

    @property
    def clock(self) -> TimeProvider:
        return self._clock

    @property
    def fee_calculator(self) -> TinkoffFeeCalculator:
        return self._fees

    def meta(self, leg: int) -> InstrumentMetadata:
        if leg not in (1, 2):
            raise ValueError("leg должен быть 1 или 2")
        return self._metadata[leg - 1]

    def open_orders(self) -> list[OrderState]:
        return [order for order in self._orders.values() if order.is_active]

    def order(self, order_id: str) -> Optional[OrderState]:
        return self._orders.get(order_id)

    def next_order_id(self) -> str:
        self._seq += 1
        return f"bota-{self._seq:08d}"

    # ------------------------------------------------------------------ события
    def _publish(self, event: ExecutionEvent) -> None:
        self.events.try_write(event)

    # ------------------------------------------------------------------ интерфейс
    @abc.abstractmethod
    async def start(self) -> None:
        """Инициализация (подписки, восстановление состояния)."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Остановка с отменой всех активных заявок."""

    @abc.abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderAck:
        """Выставить заявку."""

    @abc.abstractmethod
    async def replace_order(self, order_id: str, price: Decimal, quantity: Optional[int] = None) -> OrderAck:
        """Заменить цену/объём активной заявки."""

    @abc.abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Снять заявку."""

    @abc.abstractmethod
    def free_cash_rub(self) -> Decimal:
        """Свободные денежные средства для проверки ГО."""

    async def cancel_all(self) -> int:
        cancelled = 0
        for order in list(self.open_orders()):
            if await self.cancel_order(order.order_id):
                cancelled += 1
        return cancelled

    def on_order_book(self, leg: int, book: OrderBookSnapshot) -> None:
        """Питание шлюза стаканом (симулятору нужно, живому — нет)."""

    def on_trade(self, leg: int, trade: TradePrint) -> None:
        """Питание шлюза лентой сделок (симулятору нужно, живому — нет)."""

    async def advance_clock(self) -> None:
        """Обработка отложенных действий по времени (латентность, поллинг)."""
