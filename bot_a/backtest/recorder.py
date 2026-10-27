"""Запись live-потока L2/сделок в PostgreSQL и/или сжатые JSONL-файлы."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from bot_a.backtest.recording_format import event_to_dict
from bot_a.config import AppConfig, PairPreset
from bot_a.models.domain import MarketEvent
from bot_a.utils.time_provider import RealTimeProvider

LOGGER_NAME = "bot_a.recorder"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS order_book_ticks (
    id bigserial PRIMARY KEY,
    ts_ms bigint NOT NULL,
    instrument_uid text NOT NULL,
    leg smallint NOT NULL,
    bids jsonb NOT NULL,
    asks jsonb NOT NULL,
    last_price numeric(20, 9)
);
CREATE INDEX IF NOT EXISTS idx_order_book_ticks_ts ON order_book_ticks (ts_ms);
CREATE TABLE IF NOT EXISTS trade_prints (
    id bigserial PRIMARY KEY,
    ts_ms bigint NOT NULL,
    instrument_uid text NOT NULL,
    leg smallint NOT NULL,
    price numeric(20, 9) NOT NULL,
    quantity bigint NOT NULL,
    side text
);
CREATE INDEX IF NOT EXISTS idx_trade_prints_ts ON trade_prints (ts_ms);
"""

INSERT_BOOK_SQL = (
    "INSERT INTO order_book_ticks (ts_ms, instrument_uid, leg, bids, asks, last_price)"
    " VALUES (%s, %s, %s, %s, %s, %s)"
)
INSERT_TRADE_SQL = (
    "INSERT INTO trade_prints (ts_ms, instrument_uid, leg, price, quantity, side)"
    " VALUES (%s, %s, %s, %s, %s, %s)"
)


@dataclass(slots=True)
class RecorderStats:
    books: int = 0
    trades: int = 0
    bytes_written: int = 0
    started_ms: int = 0
    files: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "books": self.books,
            "trades": self.trades,
            "bytesWritten": self.bytes_written,
            "startedMs": self.started_ms,
            "files": list(self.files),
        }


class MarketDataRecorder:
    """Двухканальная запись: gzip-JSONL и батчевая вставка в PostgreSQL."""

    def __init__(self, config: AppConfig, preset: PairPreset, logger: Optional[logging.Logger] = None) -> None:
        self._config = config
        self._preset = preset
        self._log = logger or logging.getLogger(LOGGER_NAME)
        self._settings = config.storage
        self._file_handle: Optional[gzip.GzipFile] = None
        self._connection: Any = None
        self._books_buffer: List[Tuple[Any, ...]] = []
        self._trades_buffer: List[Tuple[Any, ...]] = []
        self._last_flush = time.monotonic()
        self._stats = RecorderStats(started_ms=int(time.time() * 1000))
        self._file_path: Optional[Path] = None

    @property
    def stats(self) -> RecorderStats:
        return self._stats

    @property
    def file_path(self) -> Optional[Path]:
        return self._file_path

    async def start(self) -> None:
        if self._settings.record_to_files:
            self._open_file()
        if self._settings.record_to_postgres:
            self._open_postgres()
        self._log.info(
            "Запись запущена: файлы=%s, postgres=%s",
            self._file_path or "выкл",
            "вкл" if self._connection is not None else "выкл",
        )

    def _open_file(self) -> None:
        directory = self._config.path(self._settings.recordings_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._file_path = directory / f"{self._preset.name}_{stamp}.jsonl.gz"
        self._file_handle = gzip.open(self._file_path, "at", encoding="utf-8", compresslevel=6)
        self._stats.files.append(str(self._file_path))

    def _open_postgres(self) -> None:
        dsn = self._settings.postgres_dsn
        if not dsn:
            self._log.info("PostgreSQL DSN не задан — пишем только в файлы")
            return
        try:
            import psycopg
        except Exception as exc:
            self._log.warning("psycopg недоступен, PostgreSQL-запись отключена: %s", exc)
            return
        try:
            self._connection = psycopg.connect(dsn, autocommit=False)
            with self._connection.cursor() as cursor:
                cursor.execute(SCHEMA_SQL)
            self._connection.commit()
            self._log.info("PostgreSQL-приёмник подключён")
        except Exception as exc:
            self._log.warning("Не удалось подключиться к PostgreSQL: %s", exc)
            self._connection = None

    def write(self, event: MarketEvent) -> None:
        payload = event_to_dict(event)
        if self._file_handle is not None:
            line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            self._file_handle.write(line)
            self._stats.bytes_written += len(line)
        if self._connection is not None:
            if event.is_book and event.book is not None:
                book = event.book
                self._books_buffer.append(
                    (
                        book.time_ms,
                        book.instrument_uid,
                        event.leg,
                        json.dumps([[str(level.price), level.quantity] for level in book.bids]),
                        json.dumps([[str(level.price), level.quantity] for level in book.asks]),
                        str(book.last_price) if book.last_price is not None else None,
                    )
                )
                self._stats.books += 1
            elif event.trade is not None:
                trade = event.trade
                self._trades_buffer.append(
                    (
                        trade.time_ms,
                        trade.instrument_uid,
                        event.leg,
                        str(trade.price),
                        trade.quantity,
                        trade.side.value if trade.side is not None else None,
                    )
                )
                self._stats.trades += 1
        elif event.is_book:
            self._stats.books += 1
        else:
            self._stats.trades += 1
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        pending = len(self._books_buffer) + len(self._trades_buffer)
        elapsed = time.monotonic() - self._last_flush
        if pending >= self._settings.recording_batch_size or elapsed >= self._settings.recording_flush_seconds:
            self.flush()

    def flush(self) -> None:
        self._last_flush = time.monotonic()
        if self._file_handle is not None:
            self._file_handle.flush()
        if self._connection is None:
            self._books_buffer.clear()
            self._trades_buffer.clear()
            return
        try:
            with self._connection.cursor() as cursor:
                if self._books_buffer:
                    cursor.executemany(INSERT_BOOK_SQL, self._books_buffer)
                if self._trades_buffer:
                    cursor.executemany(INSERT_TRADE_SQL, self._trades_buffer)
            self._connection.commit()
        except Exception as exc:
            self._log.error("Ошибка записи в PostgreSQL: %s", exc)
            try:
                self._connection.rollback()
            except Exception:
                pass
        finally:
            self._books_buffer.clear()
            self._trades_buffer.clear()

    async def close(self) -> None:
        self.flush()
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        if self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None
        self._log.info(
            "Запись остановлена: стаканов=%d, сделок=%d, файл=%s",
            self._stats.books,
            self._stats.trades,
            self._file_path,
        )


class RecordingService:
    """Подписка на gRPC-поток T-Invest и запись событий."""

    def __init__(
        self,
        config: AppConfig,
        preset: PairPreset,
        client: Any,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._preset = preset
        self._client = client
        self._log = logger or logging.getLogger(LOGGER_NAME)

    async def run(self, duration_seconds: Optional[float] = None, stop_event: Optional[asyncio.Event] = None) -> dict:
        from bot_a.broker.tinvest import TinkoffMarketDataStream

        recorder = MarketDataRecorder(self._config, self._preset, self._log)
        await recorder.start()
        clock = RealTimeProvider()
        deadline = (time.monotonic() + duration_seconds) if duration_seconds else None
        stream = TinkoffMarketDataStream(
            self._client,
            [self._preset.leg1, self._preset.leg2],
            depth=self._config.execution.order_book_depth,
            clock=clock,
        )
        try:
            async with stream:
                self._log.info(
                    "Подписка на %s / %s (стакан + сделки)",
                    self._preset.leg1.ticker,
                    self._preset.leg2.ticker,
                )
                async for event in stream.events():
                    recorder.write(event)
                    if deadline is not None and time.monotonic() >= deadline:
                        self._log.info("Достигнута заданная длительность записи")
                        break
                    if stop_event is not None and stop_event.is_set():
                        self._log.info("Получен сигнал остановки записи")
                        break
        finally:
            await recorder.close()
        return recorder.stats.to_dict()

    async def run_with_stream(self, stream, duration_seconds: Optional[float] = None, stop_event: Optional[asyncio.Event] = None) -> dict:
        """Запись из уже открытого потока (используется тестами)."""
        recorder = MarketDataRecorder(self._config, self._preset, self._log)
        await recorder.start()
        deadline = (time.monotonic() + duration_seconds) if duration_seconds else None
        try:
            async for event in stream.events():
                recorder.write(event)
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if stop_event is not None and stop_event.is_set():
                    break
        finally:
            await recorder.close()
        return recorder.stats.to_dict()
