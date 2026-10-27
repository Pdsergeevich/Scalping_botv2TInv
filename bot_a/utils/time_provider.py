"""Абстракция времени: реальные часы и симулированные часы бэктеста."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

# Москва с 2014 года живёт в фиксированном UTC+3 (без перехода на летнее время).
MSK = timezone(timedelta(hours=3), "MSK")


@runtime_checkable
class TimeProvider(Protocol):
    """Источник времени, общий для движка, шлюза исполнения и риск-контроля."""

    def now_ms(self) -> int:
        """Текущее время, миллисекунды UTC."""

    def now_msk(self) -> datetime:
        """Текущее время в часовом поясе Москвы."""


class RealTimeProvider:
    """Реальные системные часы."""

    __slots__ = ()

    def now_ms(self) -> int:
        return int(time.time() * 1000)

    def now_msk(self) -> datetime:
        return ms_to_msk(self.now_ms())


class SimulatedTimeProvider:
    """Управляемые часы для исторического реплея."""

    __slots__ = ("_now_ms",)

    def __init__(self, start_ms: int = 0) -> None:
        self._now_ms = int(start_ms)

    def now_ms(self) -> int:
        return self._now_ms

    def now_msk(self) -> datetime:
        return ms_to_msk(self._now_ms)

    def set(self, ms: int) -> None:
        if ms > self._now_ms:
            self._now_ms = int(ms)

    def advance(self, delta_ms: int) -> None:
        self._now_ms += int(delta_ms)


def ms_to_msk(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=MSK)


def msk_to_ms(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=MSK)
    return int(value.timestamp() * 1000)


def trading_day_key(ms: int) -> date:
    """Календарный торговый день в Москве (ключ для лимитов и тарифных ступеней)."""
    return ms_to_msk(ms).date()


def now_ms() -> int:
    return int(time.time() * 1000)
