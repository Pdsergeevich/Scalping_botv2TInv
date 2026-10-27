"""Живой шлюз исполнения через T-Invest gRPC ``OrdersService``."""

from __future__ import annotations

from decimal import Decimal
from typing import Optional, Sequence

from bot_a.broker.tinvest import BrokerOrderState, TinkoffOrderClient
from bot_a.economics.fees import TinkoffFeeCalculator
from bot_a.execution.gateway import (
    ExecutionEvent,
    ExecutionEventKind,
    OrderAck,
    OrderExecutionGateway,
    OrderRequest,
    OrderState,
)
from bot_a.models.domain import (
    InstrumentMetadata,
    Liquidity,
    OrderStatus,
    OrderType,
    Side,
)
from bot_a.utils.channel import Channel
from bot_a.utils.quotation import round_to_increment
from bot_a.utils.time_provider import TimeProvider

ZERO = Decimal("0")


class TinkoffLiveExecutionGateway(OrderExecutionGateway):
    """Отправляет реальные заявки и синхронизирует их состояние поллингом."""

    def __init__(
        self,
        metadata: Sequence[InstrumentMetadata],
        clock: TimeProvider,
        fee_calculator: TinkoffFeeCalculator,
        order_client: TinkoffOrderClient,
        poll_interval_ms: int = 250,
        cash_poll_interval_ms: int = 3000,
        event_sink: Optional[Channel[ExecutionEvent]] = None,
    ) -> None:
        super().__init__(metadata, clock, fee_calculator, event_sink)
        self._client = order_client
        self._poll_interval_ms = max(int(poll_interval_ms), 50)
        self._cash_poll_interval_ms = max(int(cash_poll_interval_ms), 500)
        self._last_poll_ms = 0
        self._last_cash_poll_ms = 0
        self._free_cash = Decimal(0)
        self._started = False

    async def start(self) -> None:
        self._started = True
        await self._refresh_cash()
        self._last_poll_ms = self._clock.now_ms()

    async def stop(self) -> None:
        await self.cancel_all()
        self._started = False

    async def place_order(self, request: OrderRequest) -> OrderAck:
        meta = self.meta(request.leg)
        price = request.price
        if price is not None:
            price = round_to_increment(price, meta.min_price_increment)
        order_id = self.next_order_id()
        try:
            accepted, broker_id, error = await self._client.place(
                leg=request.leg,
                side=request.side,
                order_type=request.order_type.value,
                quantity=request.quantity,
                price=price,
                order_id=order_id,
            )
        except Exception as exc:  # сетевые/валидационные ошибки брокера
            accepted, broker_id, error = False, order_id, str(exc)
        now = self._clock.now_ms()
        if not accepted:
            self._publish(
                ExecutionEvent(
                    kind=ExecutionEventKind.REJECTED,
                    order_id=order_id,
                    leg=request.leg,
                    side=request.side,
                    price=price or ZERO,
                    quantity=0,
                    remaining=request.quantity,
                    time_ms=now,
                    status=OrderStatus.REJECTED,
                    message=error,
                )
            )
            return OrderAck(order_id, request.leg, request.side, request.order_type, price, request.quantity, False, now, error)

        state = OrderState(
            order_id=broker_id or order_id,
            leg=request.leg,
            side=request.side,
            order_type=request.order_type,
            price=price,
            quantity=int(request.quantity),
            created_ms=now,
            updated_ms=now,
            client_tag=request.client_tag,
        )
        self._orders[state.order_id] = state
        self._publish(
            ExecutionEvent(
                kind=ExecutionEventKind.ACK,
                order_id=state.order_id,
                leg=state.leg,
                side=state.side,
                price=price or ZERO,
                quantity=0,
                remaining=state.quantity,
                time_ms=now,
                status=OrderStatus.NEW,
                message="accepted by broker",
            )
        )
        return OrderAck(
            order_id=state.order_id,
            leg=state.leg,
            side=state.side,
            order_type=state.order_type,
            price=price,
            quantity=state.quantity,
            accepted=True,
            time_ms=now,
        )

    async def replace_order(self, order_id: str, price: Decimal, quantity: Optional[int] = None) -> OrderAck:
        state = self._orders.get(order_id)
        now = self._clock.now_ms()
        if state is None or not state.is_active:
            return OrderAck(order_id, 0, Side.BUY, OrderType.LIMIT, price, quantity or 0, False, now, "order not active")
        meta = self.meta(state.leg)
        clean_price = round_to_increment(price, meta.min_price_increment)
        new_quantity = int(quantity) if quantity is not None else state.quantity
        try:
            ok, error = await self._client.replace(order_id, new_quantity, clean_price)
        except Exception as exc:
            ok, error = False, str(exc)
        if ok:
            state.price = clean_price
            state.quantity = max(new_quantity, state.filled_quantity)
            state.updated_ms = now
        return OrderAck(
            order_id=order_id,
            leg=state.leg,
            side=state.side,
            order_type=state.order_type,
            price=clean_price,
            quantity=new_quantity,
            accepted=ok,
            time_ms=now,
            message=error,
        )

    async def cancel_order(self, order_id: str) -> bool:
        state = self._orders.get(order_id)
        if state is None or not state.is_active:
            return False
        try:
            ok, _ = await self._client.cancel(order_id)
        except Exception:
            return False
        return ok

    def free_cash_rub(self) -> Decimal:
        return self._free_cash

    async def advance_clock(self) -> None:
        if not self._started:
            return
        now = self._clock.now_ms()
        if now - self._last_poll_ms < self._poll_interval_ms:
            return
        self._last_poll_ms = now
        await self._poll_orders()
        if now - self._last_cash_poll_ms >= self._cash_poll_interval_ms:
            await self._refresh_cash()
        if len(self._orders) > 512:
            finished = [order_id for order_id, state in self._orders.items() if not state.is_active]
            for order_id in finished:
                del self._orders[order_id]

    async def _refresh_cash(self) -> None:
        self._last_cash_poll_ms = self._clock.now_ms()
        try:
            self._free_cash = await self._client.free_cash_rub()
        except Exception:
            pass

    async def _poll_orders(self) -> None:
        for state in list(self._orders.values()):
            if not state.is_active:
                continue
            snapshot: Optional[BrokerOrderState] = await self._client.get_state(state.order_id)
            if snapshot is None:
                continue
            self._apply_snapshot(state, snapshot)

    def _apply_snapshot(self, state: OrderState, snapshot: BrokerOrderState) -> None:
        now = self._clock.now_ms()
        if snapshot.lots_requested > 0:
            state.quantity = max(state.quantity, snapshot.lots_requested)
        executed = max(snapshot.lots_executed, 0)
        delta = executed - state.filled_quantity
        if delta > 0:
            price = snapshot.executed_price if snapshot.executed_price > 0 else (state.price or ZERO)
            fee = self._fees.register_fill(price, delta, self.meta(state.leg))
            state.apply_fill(delta, price, now)
            self._publish(
                ExecutionEvent(
                    kind=ExecutionEventKind.FILL,
                    order_id=state.order_id,
                    leg=state.leg,
                    side=state.side,
                    price=price,
                    quantity=delta,
                    remaining=state.remaining,
                    time_ms=now,
                    liquidity=Liquidity.MAKER if state.order_type is OrderType.LIMIT else Liquidity.TAKER,
                    commission_rub=fee,
                    status=state.status,
                    message="broker fill",
                )
            )
        if snapshot.status is OrderStatus.CANCELLED and state.is_active:
            state.status = OrderStatus.CANCELLED
            state.updated_ms = now
            self._publish(
                ExecutionEvent(
                    kind=ExecutionEventKind.CANCELLED,
                    order_id=state.order_id,
                    leg=state.leg,
                    side=state.side,
                    price=state.price or ZERO,
                    quantity=state.filled_quantity,
                    remaining=state.remaining,
                    time_ms=now,
                    status=OrderStatus.CANCELLED,
                    message="cancelled by broker",
                )
            )
        elif snapshot.status is OrderStatus.REJECTED and state.is_active:
            state.status = OrderStatus.REJECTED
            state.updated_ms = now
            self._publish(
                ExecutionEvent(
                    kind=ExecutionEventKind.REJECTED,
                    order_id=state.order_id,
                    leg=state.leg,
                    side=state.side,
                    price=state.price or ZERO,
                    quantity=state.filled_quantity,
                    remaining=state.remaining,
                    time_ms=now,
                    status=OrderStatus.REJECTED,
                    message=snapshot.message or "rejected by broker",
                )
            )
