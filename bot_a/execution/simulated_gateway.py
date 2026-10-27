"""Реалистичный симулятор исполнения (paper-режим и бэктест).

Моделируются:
* латентность постановки/замены/снятия заявки (``SimulatedOrderLatencyMs``);
* мейкерское исполнение по стакану (встречная цена дошла до нашей) и по ленте
  сделок с учётом «очереди» (``QueueFillFactor``);
* тейкерское исполнение по VWAP доступных уровней плюс шаги проскальзывания;
* комиссия T-Invest по каждой стороне и накопление дневного оборота.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Sequence

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
    OrderBookSnapshot,
    OrderStatus,
    OrderType,
    Side,
    TradePrint,
)
from bot_a.utils.channel import Channel
from bot_a.utils.quotation import dec, round_to_increment, shift_by_steps
from bot_a.utils.time_provider import TimeProvider

ZERO = Decimal("0")
_ACTIVATE = "activate"
_CANCEL = "cancel"
_REPLACE = "replace"
_TAKER = "taker"


@dataclass(slots=True)
class _SimOrder:
    """Внутреннее состояние симулируемой заявки."""

    state: OrderState
    ready_ms: int
    active: bool = False
    traded_volume: int = 0
    filled_notional: Decimal = ZERO
    is_hedge: bool = False


class SimulatedOrderExecutionGateway(OrderExecutionGateway):
    """Симулятор исполнения с латентностью, очередью и слиппеджем."""

    def __init__(
        self,
        metadata: Sequence[InstrumentMetadata],
        clock: TimeProvider,
        fee_calculator: TinkoffFeeCalculator,
        latency_ms: int = 25,
        queue_fill_factor: float = 0.35,
        slippage_steps: int = 1,
        order_book_depth: int = 5,
        initial_cash_rub: Decimal = Decimal("300000"),
        event_sink: Optional[Channel[ExecutionEvent]] = None,
    ) -> None:
        super().__init__(metadata, clock, fee_calculator, event_sink)
        self._orders: dict[str, _SimOrder] = {}
        self._books: dict[int, OrderBookSnapshot] = {}
        self._net_qty: dict[int, int] = {1: 0, 2: 0}
        self._cash = dec(initial_cash_rub)
        self._latency_ms = max(int(latency_ms), 0)
        self._queue_factor = min(max(float(queue_fill_factor), 1e-6), 1.0)
        self._slippage_steps = max(int(slippage_steps), 0)
        self._depth = max(int(order_book_depth), 1)
        self._pending: list[tuple[int, int, str, str, Optional[Decimal], Optional[int]]] = []
        self._action_seq = itertools.count()
        self._started = False

    # ------------------------------------------------------------------ служебное
    @property
    def latency_ms(self) -> int:
        return self._latency_ms

    def _schedule(
        self,
        effective_ms: int,
        action: str,
        order_id: str,
        price: Optional[Decimal] = None,
        quantity: Optional[int] = None,
    ) -> None:
        heapq.heappush(self._pending, (int(effective_ms), next(self._action_seq), action, order_id, price, quantity))

    def _now(self) -> int:
        return self._clock.now_ms()

    def open_orders(self) -> list[OrderState]:
        return [sim.state for sim in self._orders.values() if sim.state.is_active]

    def order(self, order_id: str) -> Optional[OrderState]:
        sim = self._orders.get(order_id)
        return sim.state if sim is not None else None

    def _sim(self, order_id: str) -> Optional[_SimOrder]:
        return self._orders.get(order_id)

    def _publish_status(self, sim: _SimOrder, kind: ExecutionEventKind, message: str = "") -> None:
        state = sim.state
        self._publish(
            ExecutionEvent(
                kind=kind,
                order_id=state.order_id,
                leg=state.leg,
                side=state.side,
                price=state.price or ZERO,
                quantity=state.filled_quantity,
                remaining=state.remaining,
                time_ms=self._now(),
                liquidity=Liquidity.MAKER,
                status=state.status,
                message=message,
            )
        )

    # ------------------------------------------------------------------ интерфейс
    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        await self.cancel_all()
        self._started = False

    async def place_order(self, request: OrderRequest) -> OrderAck:
        if request.quantity <= 0:
            return OrderAck("", request.leg, request.side, request.order_type, request.price, request.quantity, False, self._now(), "quantity <= 0")
        meta = self.meta(request.leg)
        order_id = self.next_order_id()
        now = self._now()
        price = request.price
        if price is not None:
            price = round_to_increment(price, meta.min_price_increment)
        state = OrderState(
            order_id=order_id,
            leg=request.leg,
            side=request.side,
            order_type=request.order_type,
            price=price,
            quantity=int(request.quantity),
            created_ms=now,
            updated_ms=now,
            client_tag=request.client_tag,
        )
        sim = _SimOrder(state=state, ready_ms=now + self._latency_ms)
        self._orders[order_id] = sim
        self._publish(
            ExecutionEvent(
                kind=ExecutionEventKind.ACK,
                order_id=order_id,
                leg=state.leg,
                side=state.side,
                price=price or ZERO,
                quantity=0,
                remaining=state.quantity,
                time_ms=now,
                status=OrderStatus.NEW,
                message="accepted",
            )
        )
        if request.order_type in (OrderType.MARKET, OrderType.LIMIT_IOC):
            self._schedule(sim.ready_ms, _TAKER, order_id)
        else:
            self._schedule(sim.ready_ms, _ACTIVATE, order_id)
        return OrderAck(
            order_id=order_id,
            leg=state.leg,
            side=state.side,
            order_type=state.order_type,
            price=price,
            quantity=state.quantity,
            accepted=True,
            time_ms=now,
        )

    async def replace_order(self, order_id: str, price: Decimal, quantity: Optional[int] = None) -> OrderAck:
        sim = self._sim(order_id)
        now = self._now()
        if sim is None or not sim.state.is_active:
            return OrderAck(order_id, 0, Side.BUY, OrderType.LIMIT, price, quantity or 0, False, now, "order not active")
        meta = self.meta(sim.state.leg)
        clean_price = round_to_increment(price, meta.min_price_increment)
        new_qty = int(quantity) if quantity is not None else sim.state.quantity
        self._schedule(now + self._latency_ms, _REPLACE, order_id, clean_price, new_qty)
        self._publish(
            ExecutionEvent(
                kind=ExecutionEventKind.ACK,
                order_id=order_id,
                leg=sim.state.leg,
                side=sim.state.side,
                price=clean_price,
                quantity=sim.state.filled_quantity,
                remaining=max(new_qty - sim.state.filled_quantity, 0),
                time_ms=now,
                status=OrderStatus.NEW,
                message="replace scheduled",
            )
        )
        return OrderAck(
            order_id=order_id,
            leg=sim.state.leg,
            side=sim.state.side,
            order_type=sim.state.order_type,
            price=clean_price,
            quantity=new_qty,
            accepted=True,
            time_ms=now,
        )

    async def cancel_order(self, order_id: str) -> bool:
        sim = self._sim(order_id)
        if sim is None or not sim.state.is_active:
            return False
        self._schedule(self._now() + self._latency_ms, _CANCEL, order_id)
        return True

    def free_cash_rub(self) -> Decimal:
        margin = ZERO
        for leg, qty in self._net_qty.items():
            margin += self.meta(leg).position_margin_rub(qty)
        for sim in self._orders.values():
            if sim.active and sim.state.remaining > 0:
                margin += self.meta(sim.state.leg).position_margin_rub(sim.state.remaining)
        return self._cash - margin

    def net_position(self, leg: int) -> int:
        return self._net_qty.get(leg, 0)

    # ------------------------------------------------------------------ рыночные данные
    def on_order_book(self, leg: int, book: OrderBookSnapshot) -> None:
        self._books[leg] = book
        if not self._started:
            return
        for sim in self._orders.values():
            if sim.active and sim.state.leg == leg:
                self._try_passive_fill(sim)

    def on_trade(self, leg: int, trade: TradePrint) -> None:
        if not self._started:
            return
        for sim in self._orders.values():
            if not sim.active or sim.state.leg != leg:
                continue
            limit = sim.state.price
            if limit is None or sim.state.order_type is not OrderType.LIMIT:
                continue
            if trade.price != limit:
                continue
            # Мейкер стоит в очереди: чтобы исполниться, рынок должен «переварить»
            # объём в ``1 / QueueFillFactor`` раз больше нашего.
            sim.traded_volume += max(int(trade.quantity), 0)
            required = max(int(round(sim.state.quantity / self._queue_factor)), sim.state.quantity)
            if sim.traded_volume >= required:
                self._fill(sim, sim.state.remaining, limit, Liquidity.MAKER, "queue fill")

    async def advance_clock(self) -> None:
        now = self._now()
        while self._pending and self._pending[0][0] <= now:
            _, _, action, order_id, price, quantity = heapq.heappop(self._pending)
            sim = self._sim(order_id)
            if sim is None:
                continue
            if action == _CANCEL:
                self._apply_cancel(sim, now)
            elif action == _REPLACE:
                self._apply_replace(sim, price, quantity, now)
            elif action == _TAKER:
                self._apply_taker(sim, now)
            elif action == _ACTIVATE:
                self._apply_activate(sim, now)
        if len(self._orders) > 512:
            finished = [order_id for order_id, sim in self._orders.items() if not sim.state.is_active]
            for order_id in finished:
                del self._orders[order_id]

    # ------------------------------------------------------------------ внутренняя механика
    def _apply_activate(self, sim: _SimOrder, now: int) -> None:
        if not sim.state.is_active:
            return
        sim.active = True
        sim.ready_ms = now
        if not self._try_passive_fill(sim):
            self._publish_status(sim, ExecutionEventKind.ACK, "quoted")

    def _apply_cancel(self, sim: _SimOrder, now: int) -> None:
        if not sim.state.is_active:
            return
        sim.active = False
        sim.state.status = OrderStatus.CANCELLED
        sim.state.updated_ms = now
        self._publish_status(sim, ExecutionEventKind.CANCELLED, "cancelled")

    def _apply_replace(
        self,
        sim: _SimOrder,
        price: Optional[Decimal],
        quantity: Optional[int],
        now: int,
    ) -> None:
        if not sim.state.is_active:
            return
        if price is not None:
            sim.state.price = price
        if quantity is not None and quantity > sim.state.filled_quantity:
            sim.state.quantity = quantity
        sim.state.updated_ms = now
        sim.traded_volume = 0
        sim.filled_notional = ZERO
        sim.active = True
        self._try_passive_fill(sim)
        self._publish_status(sim, ExecutionEventKind.ACK, "replaced")

    def _apply_taker(self, sim: _SimOrder, now: int) -> None:
        if not sim.state.is_active:
            return
        book = self._books.get(sim.state.leg)
        if book is None:
            sim.state.status = OrderStatus.REJECTED
            self._publish_status(sim, ExecutionEventKind.REJECTED, "no market data")
            return
        filled = self._match_taker(sim, book)
        if filled == 0:
            if sim.state.order_type in (OrderType.MARKET, OrderType.LIMIT_IOC):
                sim.active = False
                sim.state.status = OrderStatus.CANCELLED
                self._publish_status(sim, ExecutionEventKind.CANCELLED, "unfilled taker")
            return
        if sim.state.remaining > 0 and sim.state.order_type in (OrderType.MARKET, OrderType.LIMIT_IOC):
            sim.active = False
            sim.state.status = OrderStatus.CANCELLED
            self._publish_status(sim, ExecutionEventKind.CANCELLED, "ioc remainder cancelled")

    def _match_taker(self, sim: _SimOrder, book: OrderBookSnapshot) -> int:
        state = sim.state
        meta = self.meta(state.leg)
        levels = book.asks if state.side is Side.BUY else book.bids
        limit = state.price
        remaining = state.remaining
        filled = 0
        notional = ZERO
        for level in levels[: self._depth]:
            if limit is not None:
                if state.side is Side.BUY and level.price > limit:
                    break
                if state.side is Side.SELL and level.price < limit:
                    break
            take = min(remaining, level.quantity)
            if take <= 0:
                continue
            notional += level.price * Decimal(take)
            filled += take
            remaining -= take
            if remaining == 0:
                break
        if filled == 0:
            return 0
        avg_price = notional / Decimal(filled)
        if self._slippage_steps and state.side is Side.BUY:
            avg_price = shift_by_steps(avg_price, self._slippage_steps, meta.min_price_increment)
        elif self._slippage_steps:
            avg_price = shift_by_steps(avg_price, -self._slippage_steps, meta.min_price_increment)
        avg_price = round_to_increment(avg_price, meta.min_price_increment)
        if limit is not None:
            if state.side is Side.BUY and avg_price > limit:
                avg_price = limit
            if state.side is Side.SELL and avg_price < limit:
                avg_price = limit
        self._fill(sim, filled, avg_price, Liquidity.TAKER, "taker")
        return filled

    def _try_passive_fill(self, sim: _SimOrder) -> bool:
        if not sim.active or sim.state.order_type is not OrderType.LIMIT:
            return False
        book = self._books.get(sim.state.leg)
        if book is None or sim.state.price is None:
            return False
        state = sim.state
        if state.side is Side.BUY:
            if not book.asks or book.asks[0].price > state.price:
                return False
            available = 0
            for level in book.asks[: self._depth]:
                if level.price > state.price:
                    break
                available += level.quantity
            quantity = min(state.remaining, available)
        else:
            if not book.bids or book.bids[0].price < state.price:
                return False
            available = 0
            for level in book.bids[: self._depth]:
                if level.price < state.price:
                    break
                available += level.quantity
            quantity = min(state.remaining, available)
        if quantity <= 0:
            return False
        self._fill(sim, quantity, state.price, Liquidity.MAKER, "book touch")
        return True

    def _fill(self, sim: _SimOrder, quantity: int, price: Decimal, liquidity: Liquidity, message: str) -> None:
        quantity = min(max(int(quantity), 0), sim.state.remaining)
        if quantity <= 0:
            return
        state = sim.state
        meta = self.meta(state.leg)
        fee = self._fees.register_fill(price, quantity, meta)
        state.apply_fill(quantity, price, self._now())
        sim.filled_notional += price * Decimal(quantity)
        self._net_qty[state.leg] = self._net_qty.get(state.leg, 0) + state.side.sign * quantity
        if state.remaining == 0:
            sim.active = False
            state.status = OrderStatus.FILLED
        self._publish(
            ExecutionEvent(
                kind=ExecutionEventKind.FILL,
                order_id=state.order_id,
                leg=state.leg,
                side=state.side,
                price=price,
                quantity=quantity,
                remaining=state.remaining,
                time_ms=self._now(),
                liquidity=liquidity,
                commission_rub=fee,
                status=state.status,
                message=message,
            )
        )

    def snapshot(self) -> dict:
        return {
            "cashRub": str(self._cash),
            "freeCashRub": str(self.free_cash_rub()),
            "netLeg1": self._net_qty.get(1, 0),
            "netLeg2": self._net_qty.get(2, 0),
            "activeOrders": len(self.open_orders()),
        }
