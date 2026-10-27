"""Бэктест по историческим свечам T-Invest с синтезом внутриминутных тиков."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from bot_a.backtest.candles import Candle, CandleSource
from bot_a.backtest.reporting import render_console_report, write_reports
from bot_a.backtest.simulation import SimulationOutcome, simulate_events
from bot_a.backtest.synthetics import (
    TICKS_PER_MINUTE,
    synthesize_close_minute,
    synthesize_minute,
    synthesize_pair_minute,
)
from bot_a.config import AppConfig, PairPreset
from bot_a.engine.fsm import StrategyParams
from bot_a.models.domain import (
    BacktestTotals,
    DailyBacktestSummary,
    ExecutionTopology,
    InstrumentMetadata,
    MarketEvent,
    TradeRecord,
)
from bot_a.utils.time_provider import ms_to_msk, now_ms

LOGGER_NAME = "bot_a.backtest"


@dataclass(frozen=True, slots=True)
class BacktestResult:
    title: str
    preset_name: str
    topology: ExecutionTopology
    params: StrategyParams
    summaries: Tuple[DailyBacktestSummary, ...]
    totals: BacktestTotals
    trades: Tuple[TradeRecord, ...]
    started_ms: int
    finished_ms: int
    candles_leg1: int
    candles_leg2: int
    open_position_at_end: bool
    report_paths: Tuple[str, ...] = ()
    events_processed: int = 0
    counters: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "preset": self.preset_name,
            "topology": self.topology.value,
            "params": self.params.to_dict(),
            "start": ms_to_msk(self.started_ms).isoformat(),
            "end": ms_to_msk(self.finished_ms).isoformat(),
            "candlesLeg1": self.candles_leg1,
            "candlesLeg2": self.candles_leg2,
            "openPositionAtEnd": self.open_position_at_end,
            "eventsProcessed": self.events_processed,
            "totals": {
                "trades": self.totals.trades,
                "profitableDays": self.totals.profitable_days,
                "totalDays": self.totals.total_days,
                "grossPnlRub": str(self.totals.gross_pnl_rub),
                "feesRub": str(self.totals.fees_rub),
                "netPnlRub": str(self.totals.net_pnl_rub),
                "winRatePct": self.totals.win_rate_pct,
                "profitFactor": self.totals.profit_factor,
                "sharpeAnnual": self.totals.sharpe_annual,
                "avgHoldSeconds": self.totals.avg_hold_seconds,
                "feeDragPct": self.totals.fee_drag_pct,
                "maxDrawdownRub": str(self.totals.max_drawdown_rub),
            },
            "counters": dict(self.counters),
            "reportPaths": list(self.report_paths),
        }


def iter_candle_events(
    meta1: InstrumentMetadata,
    meta2: InstrumentMetadata,
    candles1: Sequence[Candle],
    candles2: Sequence[Candle],
    ticks_per_minute: int = TICKS_PER_MINUTE,
    depth: int = 5,
    phase_lock: bool = True,
    synth_mode: str = "ticks",
) -> Iterator[MarketEvent]:
    """Итератор рыночных событий пары.

    ``synth_mode="ticks"`` — 12 подтиков на минуту (по умолчанию, с фазовой
    синхронизацией по ведущей ноге при ``phase_lock=True``).
    ``synth_mode="close"`` — только реальные закрытия 1-минутных баров без
    синтетических подтиков (zero-artifact режим).
    """
    by_time1 = {candle.time_ms: candle for candle in candles1}
    by_time2 = {candle.time_ms: candle for candle in candles2}
    for time_ms in sorted(set(by_time1.keys()) | set(by_time2.keys())):
        candle1 = by_time1.get(time_ms)
        candle2 = by_time2.get(time_ms)
        if synth_mode == "close":
            group: List[MarketEvent] = []
            if candle1 is not None:
                group.extend(synthesize_close_minute(1, meta1, candle1, depth))
            if candle2 is not None:
                group.extend(synthesize_close_minute(2, meta2, candle2, depth))
            group.sort(key=lambda event: (event.time_ms, 0 if event.is_book else 1))
            for event in group:
                yield event
            continue
        if phase_lock:
            for event in synthesize_pair_minute(
                1, meta1, candle1, 2, meta2, candle2, ticks_per_minute, depth
            ):
                yield event
            continue
        group = []
        if candle1 is not None:
            group.extend(synthesize_minute(1, meta1, candle1, ticks_per_minute, depth))
        if candle2 is not None:
            group.extend(synthesize_minute(2, meta2, candle2, ticks_per_minute, depth))
        group.sort(key=lambda event: (event.time_ms, 0 if event.is_book else 1))
        for event in group:
            yield event


async def simulate_on_candles(
    config: AppConfig,
    preset: PairPreset,
    candles1: Sequence[Candle],
    candles2: Sequence[Candle],
    params: StrategyParams,
    topology: ExecutionTopology,
    ticks_per_minute: int = TICKS_PER_MINUTE,
    logger: Optional[logging.Logger] = None,
    synth_mode: str = "ticks",
    phase_lock: bool = True,
    trace: Optional[Any] = None,
) -> SimulationOutcome:
    """Прогоняет единое ядро по синтезированным тикам (общий путь с live/paper)."""
    log = logger or logging.getLogger(LOGGER_NAME)
    start_ms = min(
        candles1[0].time_ms if candles1 else now_ms(),
        candles2[0].time_ms if candles2 else now_ms(),
    )
    events = iter_candle_events(
        preset.leg1,
        preset.leg2,
        candles1,
        candles2,
        ticks_per_minute,
        config.execution.order_book_depth,
        phase_lock=phase_lock,
        synth_mode=synth_mode,
    )
    return await simulate_events(config, preset, events, params, topology, start_ms, log, trace)


class HistoricalCandleBacktester:
    """Бэктест ``--mode backtest``: свечи из API/кэша + синтез тиков."""

    def __init__(
        self,
        config: AppConfig,
        preset: PairPreset,
        source: CandleSource,
        params: Optional[StrategyParams] = None,
        topology: Optional[ExecutionTopology] = None,
        ticks_per_minute: int = TICKS_PER_MINUTE,
        logger: Optional[logging.Logger] = None,
        synth_mode: str = "ticks",
        phase_lock: bool = True,
    ) -> None:
        self._config = config
        self._preset = preset
        self._source = source
        self._params = params or config.strategy
        self._topology = topology or config.execution.topology
        self._ticks = ticks_per_minute
        self._synth_mode = synth_mode
        self._phase_lock = phase_lock
        self._log = logger or logging.getLogger(LOGGER_NAME)
        self._last_candles: Tuple[List[Candle], List[Candle]] = ([], [])
        self._last_window: Tuple[int, int] = (0, 0)

    @property
    def last_candles(self) -> Tuple[List[Candle], List[Candle]]:
        """Свечи последнего прогона (нужны для графиков)."""
        return self._last_candles

    async def run(
        self,
        days: int = 7,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        interval: str = "1min",
        write_report: bool = True,
        report_stem: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> BacktestResult:
        finish = end_ms if end_ms is not None else now_ms()
        start = start_ms if start_ms is not None else finish - days * 86_400_000
        self._log.info(
            "Загрузка свечей %s / %s за %s .. %s",
            self._preset.leg1.ticker,
            self._preset.leg2.ticker,
            ms_to_msk(start).strftime("%Y-%m-%d %H:%M"),
            ms_to_msk(finish).strftime("%Y-%m-%d %H:%M"),
        )
        candles1, candles2 = await self._fetch(start, finish, interval)
        self._last_candles = (list(candles1), list(candles2))
        self._last_window = (start, finish)
        if not candles1 or not candles2:
            raise RuntimeError(
                "Не удалось загрузить свечи обеих ног. Проверьте токен, тикеры и торговые дни в периоде."
            )
        self._log.info("Свечей: %s = %d, %s = %d", self._preset.leg1.ticker, len(candles1), self._preset.leg2.ticker, len(candles2))
        if self._synth_mode == "close" and self._params.ewma_window > 480:
            self._log.warning(
                "Режим close: EwmaWindow=%d трактуется как %d минутных баров (%.1f ч). "
                "Для 1-минутных данных разумно 120–240 баров (--ewma-window 120).",
                self._params.ewma_window,
                self._params.ewma_window,
                self._params.ewma_window / 60.0,
            )
        outcome = await simulate_on_candles(
            self._config,
            self._preset,
            candles1,
            candles2,
            self._params,
            self._topology,
            self._ticks,
            self._log,
            synth_mode=self._synth_mode,
            phase_lock=self._phase_lock,
            trace=trace,
        )
        summaries = tuple(outcome.ledger.daily_summaries())
        totals = outcome.ledger.totals()
        title = f"Бэктест {self._preset.name} ({self._topology.value})"
        context = {
            "Пресет": self._preset.name,
            "Топология": self._topology.value,
            "Тариф": self._config.fees.tariff.value,
            "Z_Entry": self._params.z_entry,
            "Z_Exit": self._params.z_exit,
            "EwmaWindow": self._params.ewma_window,
            "Латентность, мс": self._config.execution.simulated_order_latency_ms,
            "QueueFillFactor": self._config.execution.queue_fill_factor,
            "Тиков на минуту": self._ticks if self._synth_mode != "close" else "— (close)",
            "Мост": f"{self._synth_mode}" + (" / phase-locked" if self._phase_lock and self._synth_mode != "close" else ""),
            "Период": f"{ms_to_msk(start):%Y-%m-%d %H:%M} .. {ms_to_msk(finish):%Y-%m-%d %H:%M} MSK",
            "Событий обработано": outcome.events_processed,
        }
        console = render_console_report(title, summaries, totals, context)
        self._log.info("\n%s", console)
        report_paths: Tuple[str, ...] = ()
        if write_report:
            stem = report_stem or f"backtest_{self._preset.name}_{self._topology.value}_{datetime.now():%Y%m%d_%H%M%S}"
            paths = write_reports(
                self._config.path(self._config.storage.reports_dir),
                stem,
                title,
                summaries,
                totals,
                context,
                outcome.ledger.trades,
                extra={"result": {"eventsProcessed": outcome.events_processed}},
            )
            report_paths = tuple(paths)
            self._log.info("Отчёты сохранены: %s", ", ".join(paths))
        return BacktestResult(
            title=title,
            preset_name=self._preset.name,
            topology=self._topology,
            params=self._params,
            summaries=summaries,
            totals=totals,
            trades=tuple(outcome.ledger.trades),
            started_ms=start,
            finished_ms=finish,
            candles_leg1=len(candles1),
            candles_leg2=len(candles2),
            open_position_at_end=outcome.open_position_at_end,
            report_paths=report_paths,
            events_processed=outcome.events_processed,
            counters=dict(getattr(outcome.engine, "counters", {}) or {}),
        )

    async def _fetch(self, start_ms: int, end_ms: int, interval: str) -> Tuple[List[Candle], List[Candle]]:
        candles1, candles2 = await asyncio.gather(
            self._source.get_candles(self._preset.leg1, start_ms, end_ms, interval),
            self._source.get_candles(self._preset.leg2, start_ms, end_ms, interval),
        )
        return list(candles1), list(candles2)
