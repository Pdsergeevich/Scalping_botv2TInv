"""Роллы continuous-серий FORTS (v4.4).

`*_CONT` собираются склейкой фронт-контрактов, и на дне переключения в цену
попадал спред ролла ΔP = P_новый − P_старый (`GAZR +357 пт`, `VTBR −575 пт`,
`SBRF −2 363 пт`). После back-adjustment в `discover_forts_universe.stitch`
цены непрерывны, но **сделка всё равно не должна пересекать переключение**
(иначе PnL считается по двум разным контрактам). Здесь читается карта роллов
и вычисляются бары-барьеры: последний бар перед роллом.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from bot_a.utils.time_provider import MSK

ROOT = Path(__file__).resolve().parents[2]
ROLLS_CACHE = ROOT / "data_cache" / "forts_rolls.json"


def load_rolls(ticker: str, path: Path | None = None) -> List[Dict[str, Any]]:
    """Список роллов инструмента: `{day, from, to, delta, mode}` (пусто = роллов нет)."""
    cache_path = path or ROLLS_CACHE
    if not cache_path.exists():
        return []
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return list(data.get(ticker) or [])


def roll_days(ticker: str, path: Path | None = None) -> List[str]:
    """Дни переключения контракта в формате `YYYYMMDD`."""
    return [str(item.get("day")) for item in load_rolls(ticker, path) if item.get("day")]


def boundary_ms(ticker: str, candles: Iterable[Any], path: Path | None = None) -> Set[int]:
    """Бары-барьеры: последний бар перед каждым роллом (по ним закрываем позицию).

    Возвращаются именно **времена баров**, а не дни: барьер срабатывает ровно на
    последнем баре уходящего контракта, поэтому новая сделка в тот же день уже
    не открывается (пересидеть ролл внутри сделки нельзя).
    """
    days = roll_days(ticker, path)
    if not days:
        return set()
    day_last: Dict[str, int] = {}
    for candle in candles:
        stamp = datetime.fromtimestamp(int(candle.time_ms) / 1000, tz=MSK).strftime("%Y%m%d")
        day_last[stamp] = max(day_last.get(stamp, 0), int(candle.time_ms))
    ordered = sorted(day_last)
    boundaries: Set[int] = set()
    for day in days:
        earlier = [stamp for stamp in ordered if stamp < day]
        if earlier:
            boundaries.add(day_last[earlier[-1]])
    return boundaries
