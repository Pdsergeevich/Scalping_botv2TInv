"""Высокоточный реплей записанных L2-стаканов и принтов сделок."""

from __future__ import annotations

import asyncio
import gzip
import heapq
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional, Protocol, Sequence, Tuple

from bot_a.backtest.historical_candles import BacktestResult
from bot_a.backtest.recording_format import book_from_row, event_from_json, trade_from_row
from bot_a.backtest.reporting import render_console_report, write_reports
from bot_a.backtest.simulation import simulate_events
from bot_a.config import AppConfig, PairPreset
from bot_a.engine.fsm import StrategyParams
from bot_a.models.domain import ExecutionTopology, MarketEvent
from bot_a.utils.time_provider import ms_to_msk, now_ms

LOGGER_NAME = "bot_a.replay"


class ReplayEventSource(Protocol):
    """Источник записанных событий."""

    def events(self, start_ms: int, end_ms: int) -> AsyncIterator[MarketEvent]:
        ...


def _merge_key(event: MarketEvent) -> Tuple[int, int]:
    return event.time_ms, 0 if event.is_book else 1


class JsonlRecordingSource:
    """Реплей из ``*.jsonl.gz`` (файл или каталог)."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        self._paths = [Path(path) for path in paths]
        if not self._paths:
            raise ValueError("Не переданы файлы записей")

    @staticmethod
    def from_path(path: str | Path) -> "JsonlRecordingSource":
        target = Path(path)
        if target.is_dir():
            files = sorted(target.glob("*.jsonl.gz")) + sorted(target.glob("*.jsonl"))
            if not files:
                raise FileNotFoundError(f"В каталоге {target} нет файлов *.jsonl.gz")
            return JsonlRecordingSource(files)
        if not target.exists():
            raise FileNotFoundError(f"Файл записи не найден: {target}")
        return JsonlRecordingSource([target])

    def _iter_file(self, path: Path) -> Iterator[MarketEvent]:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
            for line in handle:
                try:
                    event = event_from_json(line)
                except ValueError:
                    continue
                if event is not None:
                    yield event

    async def events(self, start_ms: int, end_ms: int) -> AsyncIterator[MarketEvent]:
        generators = [self._iter_file(path) for path in self._paths]
        merged = heapq.merge(*generators, key=_merge_key)
        counter = 0
        for event in merged:
            counter += 1
            if counter % 5000 == 0:
                await asyncio.sleep(0)
            if event.time_ms < start_ms:
                continue
            if event.time_ms > end_ms:
                break
            yield event


class PostgresRecordingSource:
    """Реплей из таблиц ``order_book_ticks`` / ``trade_prints``."""

    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise ValueError("Не задан PostgreSQL DSN для реплея")
        self._dsn = dsn

    def _iter_books(self, connection: Any, start_ms: int, end_ms: int) -> Iterator[MarketEvent]:
        with connection.cursor(name="books_cursor", itersize=20000) as cursor:
            cursor.execute(
                "SELECT ts_ms, instrument_uid, leg, bids, asks, last_price"
                " FROM order_book_ticks WHERE ts_ms BETWEEN %s AND %s ORDER BY ts_ms",
                (start_ms, end_ms),
            )
            for row in cursor:
                yield book_from_row(row)

    def _iter_trades(self, connection: Any, start_ms: int, end_ms: int) -> Iterator[MarketEvent]:
        with connection.cursor(name="trades_cursor", itersize=20000) as cursor:
            cursor.execute(
                "SELECT ts_ms, instrument_uid, leg, price, quantity, side"
                " FROM trade_prints WHERE ts_ms BETWEEN %s AND %s ORDER BY ts_ms",
                (start_ms, end_ms),
            )
            for row in cursor:
                yield trade_from_row(row)

    async def events(self, start_ms: int, end_ms: int) -> AsyncIterator[MarketEvent]:
        try:
            import psycopg
        except Exception as exc:  # pragma: no cover - зависит от окружения
            raise RuntimeError("Для реплея из PostgreSQL требуется psycopg") from exc
        connection = psycopg.connect(self._dsn)
        try:
            merged = heapq.merge(
                self._iter_books(connection, start_ms, end_ms),
                self._iter_trades(connection, start_ms, end_ms),
                key=_merge_key,
            )
            counter = 0
            for event in merged:
                counter += 1
                if counter % 5000 == 0:
                    await asyncio.sleep(0)
                yield event
        finally:
            connection.close()


@dataclass(frozen=True, slots=True)
class ReplayRange:
    start_ms: int
    end_ms: int

    @staticmethod
    def full() -> "ReplayRange":
        return ReplayRange(0, now_ms())

    @staticmethod
    def last_days(days: int) -> "ReplayRange":
        finish = now_ms()
        return ReplayRange(finish - days * 86_400_000, finish)

    def describe(self) -> str:
        start = ms_to_msk(self.start_ms).strftime("%Y-%m-%d %H:%M") if self.start_ms else "начало"
        end = ms_to_msk(self.end_ms).strftime("%Y-%m-%d %H:%M")
        return f"{start} .. {end} MSK"


class TickReplayBacktester:
    """Бэктест ``--mode backtest --source recordings``."""

    def __init__(
        self,
        config: AppConfig,
        preset: PairPreset,
        source: ReplayEventSource,
        params: Optional[StrategyParams] = None,
        topology: Optional[ExecutionTopology] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._preset = preset
        self._source = source
        self._params = params or config.strategy
        self._topology = topology or config.execution.topology
        self._log = logger or logging.getLogger(LOGGER_NAME)

    async def run(
        self,
        replay_range: Optional[ReplayRange] = None,
        write_report: bool = True,
        report_stem: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> BacktestResult:
        window = replay_range or ReplayRange.full()
        self._log.info(
            "Реплей %s / %s за %s",
            self._preset.leg1.ticker,
            self._preset.leg2.ticker,
            window.describe(),
        )
        events = self._source.events(window.start_ms, window.end_ms)
        outcome = await simulate_events(
            self._config,
            self._preset,
            events,
            replace(
                self._params,
                stat_sample_interval_seconds=float(
                    self._config.execution.stat_sample_interval_seconds
                ),
            ),
            self._topology,
            window.start_ms or None,
            self._log,
            trace,
            warmup_days=int(self._config.execution.bootstrap_warmup_days),
        )
        if outcome.events_processed == 0:
            raise RuntimeError("В выбранном диапазоне нет записанных событий")
        summaries = tuple(outcome.ledger.daily_summaries())
        totals = outcome.ledger.totals()
        title = f"Реплей L2 {self._preset.name} ({self._topology.value})"
        context = {
            "Пресет": self._preset.name,
            "Топология": self._topology.value,
            "Тариф": self._config.fees.tariff.value,
            "Z_Entry": self._params.z_entry,
            "Z_Exit": self._params.z_exit,
            "EwmaWindow": self._params.ewma_window,
            "Период": window.describe(),
            "Событий обработано": outcome.events_processed,
            "Латентность, мс": self._config.execution.simulated_order_latency_ms,
            "QueueFillFactor": self._config.execution.queue_fill_factor,
        }
        self._log.info("\n%s", render_console_report(title, summaries, totals, context))
        report_paths: Tuple[str, ...] = ()
        if write_report:
            stem = report_stem or f"replay_{self._preset.name}_{self._topology.value}_{datetime.now():%Y%m%d_%H%M%S}"
            paths = write_reports(
                self._config.path(self._config.storage.reports_dir),
                stem,
                title,
                summaries,
                totals,
                context,
                outcome.ledger.trades,
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
            started_ms=window.start_ms,
            finished_ms=window.end_ms,
            candles_leg1=0,
            candles_leg2=0,
            open_position_at_end=outcome.open_position_at_end,
            report_paths=report_paths,
            events_processed=outcome.events_processed,
            counters=dict(getattr(outcome.engine, "counters", {}) or {}),
        )


def build_replay_source(config: AppConfig, path: Optional[str]) -> ReplayEventSource:
    """Выбирает источник: файл/каталог записей либо PostgreSQL."""
    if path:
        return JsonlRecordingSource.from_path(path)
    dsn = config.storage.postgres_dsn
    if dsn:
        return PostgresRecordingSource(dsn)
    recordings_dir = config.path(config.storage.recordings_dir)
    return JsonlRecordingSource.from_path(recordings_dir)
