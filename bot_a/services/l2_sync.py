"""Синхронизация стаканов двух ног в REST-режимах (v2.5).

Живой REST-поллинг опрашивает ноги последовательно, поэтому возможны два
патологических случая:

1. одна нога ответила, вторая упала по таймауту — пара «половина свежая,
   половина старая», и ``Z`` считается по рассинхронизированным ценам;
2. обе ноги ответили, но биржа отдала одну из книг устаревшей.

26.10.2026 такой рассинхрон дал фантомный выброс ``Z = −2.298`` при спокойном
рынке (``VTBR/SBER`` физически не могут разойтись на 1.44 RUB за 300 мс).

Модуль отбрасывает такие тики из расчёта ``Z`` и считает ``skewSkips``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


@dataclass(slots=True)
class SkewDecision:
    """Результат проверки пары стаканов."""

    ok: bool
    reason: str = ""
    skew_ms: float = 0.0


@dataclass(slots=True)
class PairBookSyncer:
    """Решает, можно ли кормить ядро парой стаканов из одного цикла опроса."""

    interval_ms: int = 300
    skew_factor: float = 1.5
    stale_cycles: int = 8
    skew_skips: int = 0
    last_skew_ms: float = 0.0
    _signature: Dict[int, int] = field(default_factory=dict)
    _repeat: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.interval_ms = max(int(self.interval_ms), 50)
        self.stale_cycles = max(int(self.stale_cycles), 2)

    @property
    def skew_limit_ms(self) -> float:
        """Предел расхождения: 1.5 интервала опроса (450 мс при опросе 300 мс)."""
        return self.interval_ms * float(self.skew_factor)

    def check(
        self,
        fresh: Dict[int, Any],
        fetched_at: Dict[int, float],
        failed: Iterable[int] = (),
    ) -> SkewDecision:
        """``fresh`` — книги этого цикла; ``fetched_at`` — метки времени, мс."""
        failed_legs = sorted(int(item) for item in failed)
        if failed_legs:
            return self._reject(f"нога {failed_legs[0]} не ответила")
        if 1 not in fresh or 2 not in fresh:
            missing = 1 if 1 not in fresh else 2
            return self._reject(f"нет стакана ноги {missing}")

        exchange_skew = abs(int(fresh[1].time_ms) - int(fresh[2].time_ms))
        wall_skew = abs(float(fetched_at.get(1, 0.0)) - float(fetched_at.get(2, 0.0)))
        limit = self.skew_limit_ms
        if exchange_skew > limit:
            return self._reject(f"биржевые метки разошлись на {exchange_skew:.0f} мс", exchange_skew)
        if wall_skew > limit:
            return self._reject(f"запросы разошлись на {wall_skew:.0f} мс", wall_skew)
        for leg in (1, 2):
            book = fresh[leg]
            if not book.bids or not book.asks:
                # После закрытия сессии биржа отдаёт пустой стакан: считать по нему Z нельзя.
                return self._reject(f"пустой стакан ноги {leg}")

        frozen = self._frozen_leg(fresh)
        if frozen is not None:
            return self._reject(f"стакан ноги {frozen} не меняется, пока вторая двигается")
        return SkewDecision(ok=True, skew_ms=max(float(exchange_skew), wall_skew))

    def _frozen_leg(self, fresh: Dict[int, Any]) -> Optional[int]:
        """Нога, чья книга не менялась `stale_cycles` циклов, пока вторая менялась."""
        signatures: Dict[int, int] = {}
        for leg, book in fresh.items():
            best_bid = str(book.bids[0].price) if book.bids else ""
            best_ask = str(book.asks[0].price) if book.asks else ""
            signatures[leg] = hash(
                (int(book.time_ms), len(book.bids), len(book.asks), best_bid, best_ask)
            )
        for leg, signature in signatures.items():
            if self._signature.get(leg) == signature:
                self._repeat[leg] = self._repeat.get(leg, 0) + 1
            else:
                self._repeat[leg] = 0
            self._signature[leg] = signature
        for leg in (1, 2):
            other = 2 if leg == 1 else 1
            if self._repeat.get(leg, 0) >= self.stale_cycles and self._repeat.get(other, 0) == 0:
                return leg
        return None

    def _reject(self, reason: str, skew_ms: Optional[float] = None) -> SkewDecision:
        self.skew_skips += 1
        if skew_ms is not None:
            self.last_skew_ms = float(skew_ms)
        return SkewDecision(ok=False, reason=reason, skew_ms=float(skew_ms or 0.0))
