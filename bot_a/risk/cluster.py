"""Портфельный координатор кластеров «общая нога + одно направление» (v2.7).

02.11.2026 `IRAO_SBER`, `VTBR_SBER` и `AFKS_SBER` вошли в `buy` с интервалом 13 минут:
`SBER` пошёл против всех сателлитов сразу, и один рыночный импульс превратился в три
коррелированных убытка (−40.76 ₽). Три «независимых» сигнала были одной ставкой,
повторённой трижды.

Координатор ограничивает число одновременных однонаправленных позиций в кластере
пар с общей ногой (`Leg2 = SBER`). Ограничение настраивается (`1`, `2` или без лимита).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple


@dataclass(slots=True)
class ClusterState:
    """Кто сейчас держит позицию/котировку в кластере."""

    pair: str
    direction: str


@dataclass(slots=True)
class ClusterRiskCoordinator:
    """Разрешает или запрещает новый вход с учётом уже открытых позиций кластера.

    ``max_same_direction = 0`` — лимита нет; `1` — не больше одной позиции в одну
    сторону внутри кластера; `2` — не больше двух.
    """

    max_same_direction: int = 1
    cluster_key_leg: int = 2
    skips: int = 0
    skip_reasons: Dict[str, int] = field(default_factory=dict)
    _active: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return int(self.max_same_direction) > 0

    def _key(self, leg2_ticker: str) -> str:
        return f"{leg2_ticker.upper()}"

    def open_positions(self, leg2_ticker: str, direction: str) -> int:
        key = self._key(leg2_ticker)
        return sum(
            1
            for held_direction, held_cluster in self._active.values()
            if held_cluster == key and held_direction == direction
        )

    def register(self, pair: str, leg2_ticker: str, direction: str) -> None:
        """Помечает пару как держащую направление в кластере."""
        self._active[pair] = (str(direction).upper(), self._key(leg2_ticker))

    def release(self, pair: str) -> None:
        self._active.pop(pair, None)

    def reset(self) -> None:
        self._active.clear()

    def allows(self, pair: str, leg2_ticker: str, direction: str) -> bool:
        """Можно ли паре открывать новое направление в этом кластере."""
        if not self.enabled:
            return True
        direction = str(direction).upper()
        key = self._key(leg2_ticker)
        same = sum(
            1
            for held_direction, held_cluster in self._active.values()
            if held_cluster == key and held_direction == direction
        )
        return same < int(self.max_same_direction)

    def check(self, pair: str, leg2_ticker: str, direction: str) -> bool:
        """``allows`` + счётчик отсечек (для дашборда и отчётов)."""
        if self.allows(pair, leg2_ticker, direction):
            return True
        self.skips += 1
        key = f"{leg2_ticker.upper()}:{str(direction).upper()}"
        self.skip_reasons[key] = self.skip_reasons.get(key, 0) + 1
        return False

    def active_pairs(self) -> Sequence[str]:
        return tuple(self._active)

    def snapshot(self) -> Iterable[Tuple[str, str, str]]:
        return tuple((pair, direction, key) for pair, (direction, key) in self._active.items())


def cluster_filter_trades(
    trades_by_pair: Dict[str, Sequence[dict]],
    pairs_leg2: Dict[str, str],
    max_same_direction: int = 1,
) -> Tuple[Dict[str, Sequence[dict]], int]:
    """Групповой фильтр для бэктестов: гасит коррелированные входы постфактум.

    ``trades_by_pair`` — сделки вида ``{"open_ms", "close_ms", "side", ...}``.
    Обходим все сделки в хронологическом порядке и оставляем сделку только если
    в кластере (общая ``Leg2``) в этот момент меньше ``max_same_direction``
    открытых позиций в ту же сторону. Возвращает оставшиеся сделки и число отсечек.
    """
    limit = int(max_same_direction)
    if limit <= 0:
        return {pair: list(items) for pair, items in trades_by_pair.items()}, 0
    events: list[Tuple[int, int, str, dict]] = []
    for pair, items in trades_by_pair.items():
        for trade in items:
            side = str(trade.get("side", "")).upper()
            events.append((int(trade["open_ms"]), 1, pair, trade, side))
            events.append((int(trade["close_ms"]), 0, pair, trade, side))
    events.sort(key=lambda item: (item[0], item[1]))
    active: Dict[str, list[Tuple[str, str]]] = {}
    kept: Dict[str, list[dict]] = {pair: [] for pair in trades_by_pair}
    skipped = 0
    for _ts, kind, pair, trade, side in events:
        cluster = str(pairs_leg2.get(pair, "")).upper()
        if kind == 0:  # закрытие
            held = active.get(cluster)
            if held:
                for index, (_p, s) in enumerate(held):
                    if _p == pair and s == side:
                        held.pop(index)
                        break
                if not held:
                    active.pop(cluster, None)
            continue
        held = active.setdefault(cluster, [])
        same = sum(1 for _p, s in held if s == side)
        if same >= limit:
            skipped += 1
            continue
        held.append((pair, side))
        kept[pair].append(trade)
    return kept, skipped
