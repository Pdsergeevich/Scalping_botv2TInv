"""Единый цикл прокрутки событий через ядро (бэктест и реплей)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, Dict, Iterable, Optional

from bot_a.config import AppConfig, PairPreset
from bot_a.economics.fees import TinkoffFeeCalculator
from bot_a.engine.factory import create_engine, create_guard
from bot_a.engine.fsm import StrategyParams
from bot_a.engine.ledger import PortfolioLedger
from bot_a.execution.simulated_gateway import SimulatedOrderExecutionGateway
from bot_a.models.domain import ExecutionTopology, MarketEvent, OrderBookSnapshot
from bot_a.utils.time_provider import SimulatedTimeProvider


@dataclass(slots=True)
class SimulationOutcome:
    ledger: PortfolioLedger
    engine: object
    open_position_at_end: bool
    events_processed: int


async def _as_async_stream(events: Iterable[MarketEvent] | AsyncIterator[MarketEvent]) -> AsyncIterator[MarketEvent]:
    """Приводит синхронный/асинхронный поток событий к единому асинхронному виду."""
    if hasattr(events, "__aiter__"):
        async for event in events:  # type: ignore[union-attr]
            yield event
        return
    for event in events:  # type: ignore[union-attr]
        yield event


async def simulate_events(
    config: AppConfig,
    preset: PairPreset,
    events: Any,
    params: StrategyParams,
    topology: ExecutionTopology,
    start_ms: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    trace: Optional[Any] = None,
    warmup_days: int = 0,
) -> SimulationOutcome:
    """Прокручивает поток событий через то же ядро, что используется в live/paper.

    ``trace`` — необязательный ``TraceRecorder`` для графиков: только читает метрики
    и не влияет на состояние ядра. ``warmup_days > 0`` прогревает статистику
    историей MOEX ISS перед прокруткой (нужно для коротких L2-записей: окно
    ``ewmaWindow`` в 2400 тиков иначе не набирается).
    """
    log = logger or logging.getLogger("bot_a.backtest")
    beta_mode = preset.beta_mode if params.beta_mode in ("", "auto") else params.beta_mode
    effective_params = replace(params, beta=float(preset.beta), beta_mode=beta_mode)
    leg1, leg2 = preset.leg1, preset.leg2

    stream = _as_async_stream(events)
    first_event = await anext(stream, None)
    if first_event is None:
        ledger = PortfolioLedger(config.execution.paper_initial_cash_rub)
        return SimulationOutcome(ledger=ledger, engine=None, open_position_at_end=False, events_processed=0)

    clock = SimulatedTimeProvider(first_event.time_ms if start_ms is None else start_ms)
    fee_calculator = TinkoffFeeCalculator(config.fees.tariff, clock=clock)
    gateway = SimulatedOrderExecutionGateway(
        metadata=[leg1, leg2],
        clock=clock,
        fee_calculator=fee_calculator,
        latency_ms=config.execution.simulated_order_latency_ms,
        queue_fill_factor=config.execution.queue_fill_factor,
        slippage_steps=config.execution.slippage_steps,
        order_book_depth=config.execution.order_book_depth,
        initial_cash_rub=config.execution.paper_initial_cash_rub,
    )
    guard = create_guard(config, [leg1, leg2])
    ledger = PortfolioLedger(config.execution.paper_initial_cash_rub)
    engine = create_engine(
        [leg1, leg2],
        effective_params,
        topology,
        gateway,
        fee_calculator,
        clock,
        guard,
        ledger,
        log,
    )
    await gateway.start()

    if warmup_days > 0:
        from bot_a.services.bootstrap import bootstrap_warmup

        anchor = first_event.time_ms if start_ms is None else start_ms
        try:
            await bootstrap_warmup(
                config,
                preset,
                engine,
                days=int(warmup_days),
                end_ms=anchor - 1,
                logger=log,
                clock=clock,
            )
        except Exception as exc:  # прогрев не должен ломать реплей
            log.warning("Прогрев перед реплеем не удался (%s): %s", type(exc).__name__, exc)
            engine.set_bootstrap(False)

    last_books: Dict[int, OrderBookSnapshot] = {}
    processed = 0
    async for event in _chain_first(first_event, stream):
        clock.set(event.time_ms)
        await engine.on_clock()
        if event.is_book and event.book is not None:
            last_books[event.leg] = event.book
            await engine.on_order_book(event.leg, event.book)
        elif event.trade is not None:
            await engine.on_trade(event.leg, event.trade)
        if trace is not None:
            trace.record(engine.metrics, engine.state.value)
        processed += 1
        if processed % 5000 == 0:
            await asyncio.sleep(0)

    open_position = await _finalize(engine, clock, last_books)
    await engine.shutdown(flatten=False)
    await gateway.stop()
    return SimulationOutcome(
        ledger=ledger,
        engine=engine,
        open_position_at_end=open_position,
        events_processed=processed,
    )


async def _chain_first(first: MarketEvent, rest: AsyncIterator[MarketEvent]) -> AsyncIterator[MarketEvent]:
    yield first
    async for event in rest:
        yield event



async def _finalize(engine, clock: SimulatedTimeProvider, last_books: Dict[int, OrderBookSnapshot]) -> bool:
    """Пытается закрыть позицию по последнему стакану после конца истории."""
    if engine.position is None:
        return False
    end_ms = clock.now_ms()
    for attempt in range(1, 4):
        clock.set(end_ms + attempt * 1500)
        for leg, book in last_books.items():
            shifted = replace(book, time_ms=clock.now_ms())
            await engine.on_order_book(leg, shifted)
        await engine.on_clock()
        if engine.position is None:
            return False
    return True
