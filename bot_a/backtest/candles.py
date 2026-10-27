"""Загрузка исторических свечей и локальный кэш.

Кэш организован по дням: повторные прогоны бэктеста и оптимизации
не обращаются к API за уже скачанными данными.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence

from bot_a.broker import tinvest as broker
from bot_a.models.domain import InstrumentMetadata
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import MSK, ms_to_msk


@dataclass(frozen=True, slots=True)
class Candle:
    """Минутная (или иная) свеча в нормализованном виде."""

    time_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time_ms,
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": self.volume,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Candle":
        return Candle(
            time_ms=int(data["time"]),
            open=dec(data["open"]),
            high=dec(data["high"]),
            low=dec(data["low"]),
            close=dec(data["close"]),
            volume=int(data.get("volume", 0) or 0),
        )


class CandleSource(Protocol):
    """Источник свечей (API T-Invest или тестовый)."""

    async def get_candles(
        self,
        meta: InstrumentMetadata,
        start_ms: int,
        end_ms: int,
        interval: str = "1min",
    ) -> List[Candle]:
        ...


class InMemoryCandleSource:
    """Свечи из памяти — используется тестами и повторными прогонами."""

    def __init__(self, data: Dict[str, Sequence[Candle]]) -> None:
        self._data = {key: list(value) for key, value in data.items()}

    async def get_candles(
        self,
        meta: InstrumentMetadata,
        start_ms: int,
        end_ms: int,
        interval: str = "1min",
    ) -> List[Candle]:
        key = meta.ticker or meta.instrument_uid
        candles = self._data.get(key, [])
        return [candle for candle in candles if start_ms <= candle.time_ms <= end_ms]


class TInvestCandleSource:
    """Свечи через ``MarketDataService.GetCandles`` с дневным кэшем на диске."""

    def __init__(self, client: Any, cache_dir: str | Path, interval: str = "1min") -> None:
        self._client = client
        self._cache_dir = Path(cache_dir)
        self._interval = interval
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, meta: InstrumentMetadata, day: datetime) -> Path:
        instrument_key = meta.instrument_uid or f"{meta.ticker}_{meta.class_code}"
        return self._cache_dir / f"{instrument_key}_{self._interval}_{day.strftime('%Y%m%d')}.json"

    async def get_candles(
        self,
        meta: InstrumentMetadata,
        start_ms: int,
        end_ms: int,
        interval: str = "1min",
    ) -> List[Candle]:
        interval = interval or self._interval
        if not meta.instrument_uid:
            raise RuntimeError(
                f"Для инструмента {meta.ticker} не разрешён InstrumentUid — выполните --mode preflight"
            )
        first_day = ms_to_msk(start_ms).date()
        last_day = ms_to_msk(end_ms).date()
        candles: List[Candle] = []
        day = first_day
        while day <= last_day:
            candles.extend(await self._load_day(meta, day, interval))
            day += timedelta(days=1)
        result = [candle for candle in candles if start_ms <= candle.time_ms <= end_ms]
        result.sort(key=lambda item: item.time_ms)
        return result

    async def _load_day(self, meta: InstrumentMetadata, day, interval: str) -> List[Candle]:
        path = self._cache_path(meta, day)
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                return [Candle.from_dict(item) for item in raw]
            except (json.JSONDecodeError, KeyError, ValueError):
                path.unlink(missing_ok=True)
        day_start = datetime.combine(day, time.min, tzinfo=MSK)
        day_end = datetime.combine(day, time.max, tzinfo=MSK)
        raw_candles = await broker.get_candles(
            self._client,
            meta.instrument_uid,
            day_start,
            day_end,
            interval=interval,
        )
        path.write_text(json.dumps(raw_candles, ensure_ascii=False), encoding="utf-8")
        return [Candle.from_dict(item) for item in raw_candles]
