"""Paper-торговля на реальных данных MOEX ISS (без токена и без денег).

Это тот же путь, что и ``--mode paper``: то же ядро, тот же симулятор исполнения,
тот же тариф и те же окна входа. Отличие только в источнике событий: вместо
gRPC-потока T-Invest проигрываются реальные 1-минутные свечи MOEX ISS
(12 подтиков на минуту), а перед сессией выполняется прогрев ``bootstrap_warmup``.

Результат — обычный ``BacktestResult`` (сделки, дневные итоги), траектория метрик
для графиков и отчёт о прогреве.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple

from bot_a.backtest.candles import Candle
from bot_a.backtest.historical_candles import BacktestResult, iter_candle_events
from bot_a.backtest.iss_source import IssCandleSource
from bot_a.config import AppConfig, PairPreset
from bot_a.economics.fees import TinkoffFeeCalculator
from bot_a.engine.factory import create_engine, create_guard
from bot_a.engine.fsm import StrategyParams
from bot_a.engine.ledger import PortfolioLedger
from bot_a.execution.simulated_gateway import SimulatedOrderExecutionGateway
from bot_a.models.domain import ExecutionTopology
from bot_a.utils.time_provider import MSK, SimulatedTimeProvider, msk_to_ms

LOGGER_NAME = "bot_a.paper_replay"


class PaperReplayService:
    """Проигрывает реальные свечи ISS через боевое ядро с paper-исполнением."""

    def __init__(
        self,
        config: AppConfig,
        preset: PairPreset,
        source: Optional[IssCandleSource] = None,
        params: Optional[StrategyParams] = None,
        topology: Optional[ExecutionTopology] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._preset = preset
        self._source = source or IssCandleSource(config.path(config.storage.data_cache_dir))
        self._params = replace(
            params or config.strategy,
            beta=float(preset.beta),
            beta_mode=(
                preset.beta_mode
                if (params or config.strategy).beta_mode in ("", "auto")
                else (params or config.strategy).beta_mode
            ),
            stat_sample_interval_seconds=float(config.execution.stat_sample_interval_seconds),
        )
        self._topology = topology or config.execution.topology
        self._log = logger or logging.getLogger(LOGGER_NAME)

    async def run(
        self,
        start_ms: int,
        end_ms: int,
        warmup_days: int = 5,
        trace: Optional[Any] = None,
        progress_every: int = 200_000,
    ) -> Tuple[BacktestResult, Dict[str, Any], Tuple[List[Candle], List[Candle]]]:
        clock = SimulatedTimeProvider(start_ms)
        fee_calculator = TinkoffFeeCalculator(self._config.fees.tariff, clock=clock)
        gateway = SimulatedOrderExecutionGateway(
            metadata=[self._preset.leg1, self._preset.leg2],
            clock=clock,
            fee_calculator=fee_calculator,
            latency_ms=self._config.execution.simulated_order_latency_ms,
            queue_fill_factor=self._config.execution.queue_fill_factor,
            slippage_steps=self._config.execution.slippage_steps,
            order_book_depth=self._config.execution.order_book_depth,
            initial_cash_rub=self._config.execution.paper_initial_cash_rub,
        )
        guard = create_guard(self._config, [self._preset.leg1, self._preset.leg2])
        ledger = PortfolioLedger(self._config.execution.paper_initial_cash_rub)
        engine = create_engine(
            [self._preset.leg1, self._preset.leg2],
            self._params,
            self._topology,
            gateway,
            fee_calculator,
            clock,
            guard,
            ledger,
            self._log,
        )
        await gateway.start()

        warmup: Dict[str, Any] = {}
        if warmup_days > 0:
            from bot_a.services.bootstrap import bootstrap_warmup

            report = await bootstrap_warmup(
                self._config,
                self._preset,
                engine,
                source=self._source,
                days=warmup_days,
                end_ms=start_ms - 1,
                logger=self._log,
                clock=clock,
            )
            warmup = report.to_dict()

        self._log.info(
            "Paper на реальных данных: %s с %s по %s MSK",
            self._preset.name,
            datetime.fromtimestamp(start_ms / 1000, tz=MSK).strftime("%d.%m %H:%M"),
            datetime.fromtimestamp(end_ms / 1000, tz=MSK).strftime("%d.%m %H:%M"),
        )
        candles1, candles2 = await asyncio.gather(
            self._source.get_candles(self._preset.leg1, start_ms, end_ms, "1min"),
            self._source.get_candles(self._preset.leg2, start_ms, end_ms, "1min"),
        )
        candles1, candles2 = list(candles1), list(candles2)
        if not candles1 or not candles2:
            raise RuntimeError("Нет свечей MOEX ISS за выбранный период — проверьте даты и кэш")

        events = iter_candle_events(
            self._preset.leg1,
            self._preset.leg2,
            candles1,
            candles2,
            depth=self._config.execution.order_book_depth,
            phase_lock=True,
            synth_mode="ticks",
        )
        processed = 0
        for event in events:
            clock.set(event.time_ms)
            await engine.on_clock()
            if event.is_book and event.book is not None:
                await engine.on_order_book(event.leg, event.book)
            elif event.trade is not None:
                await engine.on_trade(event.leg, event.trade)
            if trace is not None:
                trace.record(engine.metrics, engine.state.value)
            processed += 1
            if processed % progress_every == 0:
                snapshot = engine.snapshot()
                self._log.info(
                    "Статус (paper): %s | Z=%.2f | сделок=%d | net=%s RUB",
                    snapshot["state"],
                    snapshot["zscore"],
                    snapshot["trades"],
                    snapshot["netPnlRub"],
                )
                await asyncio.sleep(0)

        await engine.shutdown(flatten=self._config.risk.flatten_on_shutdown)
        for _ in range(4):
            if engine.position is None:
                break
            clock.advance(30_000)
            await engine.on_clock()
        await gateway.stop()

        summaries = tuple(ledger.daily_summaries())
        totals = ledger.totals()
        result = BacktestResult(
            title=f"Paper на реальных данных {self._preset.name} ({self._topology.value})",
            preset_name=self._preset.name,
            topology=self._topology,
            params=self._params,
            summaries=summaries,
            totals=totals,
            trades=tuple(ledger.trades),
            started_ms=start_ms,
            finished_ms=end_ms,
            candles_leg1=len(candles1),
            candles_leg2=len(candles2),
            open_position_at_end=engine.position is not None,
            report_paths=(),
            events_processed=processed,
            counters=dict(engine.counters),
        )
        return result, warmup, (candles1, candles2)


def session_window(date: str) -> Tuple[int, int]:
    """Границы торговой сессии выбранного дня (MSK), удобно для CLI."""
    day = datetime.fromisoformat(date).date()
    start = msk_to_ms(datetime.combine(day, time(9, 50, tzinfo=MSK)))
    end = msk_to_ms(datetime.combine(day, time(23, 50, tzinfo=MSK)))
    return start, end
