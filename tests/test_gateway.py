"""Проверки симулятора исполнения: латентность, мейкер/тейкер, очередь, комиссии."""

import asyncio
from decimal import Decimal

from bot_a.economics.fees import Tariff, TinkoffFeeCalculator
from bot_a.execution.gateway import ExecutionEventKind, OrderRequest
from bot_a.execution.simulated_gateway import SimulatedOrderExecutionGateway
from bot_a.models.domain import OrderType, Side
from bot_a.utils.time_provider import SimulatedTimeProvider
from tests.helpers import make_book, make_meta, make_trade


def _build(latency: int = 50, slippage: int = 0, queue: float = 0.35, cash: str = "1000000"):
    leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01")
    leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01")
    clock = SimulatedTimeProvider(0)
    fee = TinkoffFeeCalculator(Tariff.TRADER, clock=clock)
    gateway = SimulatedOrderExecutionGateway(
        metadata=[leg1, leg2],
        clock=clock,
        fee_calculator=fee,
        latency_ms=latency,
        queue_fill_factor=queue,
        slippage_steps=slippage,
        order_book_depth=5,
        initial_cash_rub=Decimal(cash),
    )
    return gateway, clock, fee


def _drain(gateway):
    events = []
    while True:
        event = gateway.events.try_read()
        if event is None:
            return events
        events.append(event)


def test_latency_prevents_immediate_maker_fill():
    async def scenario():
        gateway, clock, _ = _build(latency=50)
        await gateway.start()
        clock.set(1000)
        gateway.on_order_book(1, make_book("100.00", 1000))
        ack = await gateway.place_order(OrderRequest(1, Side.BUY, OrderType.LIMIT, 1, Decimal("100.00")))
        assert ack.accepted
        _drain(gateway)

        clock.set(1010)
        gateway.on_order_book(1, make_book("99.99", 1010))  # ask 100.00 <= нашей цене
        assert not [event for event in _drain(gateway) if event.kind is ExecutionEventKind.FILL]

        clock.set(1060)
        await gateway.advance_clock()
        fills = [event for event in _drain(gateway) if event.kind is ExecutionEventKind.FILL]
        assert len(fills) == 1
        assert fills[0].price == Decimal("100.00")
        assert fills[0].liquidity.value == "maker"

    asyncio.run(scenario())


def test_passive_fill_by_book_touch():
    async def scenario():
        gateway, clock, _ = _build(latency=0)
        await gateway.start()
        clock.set(1000)
        gateway.on_order_book(1, make_book("100.00", 1000))
        await gateway.place_order(OrderRequest(1, Side.BUY, OrderType.LIMIT, 2, Decimal("100.00")))
        await gateway.advance_clock()
        _drain(gateway)
        gateway.on_order_book(1, make_book("99.98", 2000))
        fills = [event for event in _drain(gateway) if event.kind is ExecutionEventKind.FILL]
        assert fills, "пассивная заявка должна исполниться при касании стакана"
        assert fills[0].price == Decimal("100.00")
        assert fills[0].remaining == 0

    asyncio.run(scenario())


def test_queue_model_fill_by_trade_prints():
    async def scenario():
        gateway, clock, _ = _build(latency=0, queue=0.35)
        await gateway.start()
        clock.set(1000)
        gateway.on_order_book(1, make_book("100.00", 1000))
        await gateway.place_order(OrderRequest(1, Side.BUY, OrderType.LIMIT, 1, Decimal("100.00")))
        await gateway.advance_clock()
        _drain(gateway)
        gateway.on_trade(1, make_trade("100.00", 1100, quantity=2))
        assert not [event for event in _drain(gateway) if event.kind is ExecutionEventKind.FILL]
        gateway.on_trade(1, make_trade("100.00", 1200, quantity=1))
        fills = [event for event in _drain(gateway) if event.kind is ExecutionEventKind.FILL]
        assert fills and fills[0].liquidity.value == "maker"
        assert fills[0].quantity == 1

    asyncio.run(scenario())


def test_taker_fill_with_slippage_and_commission():
    async def scenario():
        gateway, clock, fee = _build(latency=0, slippage=2)
        await gateway.start()
        clock.set(1000)
        gateway.on_order_book(1, make_book("100.00", 1000))
        await gateway.place_order(OrderRequest(1, Side.BUY, OrderType.MARKET, 1, None))
        await gateway.advance_clock()
        events = _drain(gateway)
        fills = [event for event in events if event.kind is ExecutionEventKind.FILL]
        assert fills and fills[0].liquidity.value == "taker"
        assert fills[0].price == Decimal("100.03")  # ask 100.01 + 2 шага
        assert fills[0].commission_rub == Decimal("100.03") * Decimal("0.0004")
        assert fee.daily_turnover_rub == Decimal("100.03")
        assert gateway.net_position(1) == 1
        assert gateway.free_cash_rub() == Decimal("1000000") - Decimal("1000")

    asyncio.run(scenario())


def test_cancel_is_delayed_by_latency():
    async def scenario():
        gateway, clock, _ = _build(latency=50)
        await gateway.start()
        clock.set(1000)
        gateway.on_order_book(1, make_book("100.00", 1000))
        ack = await gateway.place_order(OrderRequest(1, Side.SELL, OrderType.LIMIT, 1, Decimal("100.01")))
        assert await gateway.cancel_order(ack.order_id)
        clock.set(1010)
        await gateway.advance_clock()
        assert not [event for event in _drain(gateway) if event.kind is ExecutionEventKind.CANCELLED]
        clock.set(1050)
        await gateway.advance_clock()
        assert [event for event in _drain(gateway) if event.kind is ExecutionEventKind.CANCELLED]
        assert not gateway.open_orders()

    asyncio.run(scenario())
