"""Базовые активы FORTS и резолв живого фронт-контракта (v4.1).

Две задачи:

1. **Дедупликация базовых активов.** `CNYRUBF` (перпетуал) и `CNY_CONT` (квартальная
   склейка) — один и тот же юань; держать оба в корзине бессмысленно: они делят
   слоты и маржу, а корреляция между ними ≈ 1. `underlying_family()` сводит тикеры
   к семейству базового актива так, чтобы на семейство оставался один инструмент.
2. **Резолв `*_CONT` → живой контракт.** Continuous-серия — это склейка фронт-месяцев
   для бэктеста. Для `paper`/`live` нужен реально торгуемый контракт семейства:
   `resolve_front_contract()` берёт карту `front_by_month` из
   `data_cache/forts_universe.json`, а если она устарела — живой список ISS.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Семейства базовых активов: тикер → семейство. Всё, чего нет в таблице,
# считается своим собственным семейством (имя тикера).
FAMILY_BY_TICKER: Dict[str, str] = {
    # юань
    "CNYRUBF": "CNY",
    "CNY_CONT": "CNY",
    "CRZ6": "CNY",
    "CRU6": "CNY",
    "CRM6": "CNY",
    # доллар
    "USDRUBF": "USD",
    "SIZ6": "USD",
    "SI_CONT": "USD",
    # евро
    "EURRUBF": "EUR",
    "EUCONT": "EUR",
    "EU_CONT": "EUR",
    "EUM6": "EUR",
    "EUU6": "EUR",
    "EUZ6": "EUR",
    # золото
    "GLDRUBF": "GOLD",
    "GOLDM_CONT": "GOLD",
    "GL_CONT": "GOLD",
    "GOLD_CONT": "GOLD",
    # серебро
    "SLVRUBF": "SILVER",
    "SILV_CONT": "SILVER",
    "SILVM_CONT": "SILVER",
    # газпром
    "GAZPF": "GAZP",
    "GAZR_CONT": "GAZP",
    # индекс
    "IMOEXF": "INDEX",
    "MXI_CONT": "INDEX",
    # нефть
    "BRM_CONT": "BRENT",
    # прочее
    "COCOA_CONT": "COCOA",
    "SPYF_CONT": "SP500",
    "NASD_CONT": "NASDAQ",
    "VTBR_CONT": "VTBR",
    "SBRF_CONT": "SBER",
    "LKOH_CONT": "LKOH",
    "ROSN_CONT": "ROSN",
    "YDEX_CONT": "YDEX",
    "OZON_CONT": "OZON",
    "PLZLM_CONT": "PLZL",
    "RGBI_CONT": "RGBI",
    "NG_CONT": "NG",
    "NGM_CONT": "NGM",
    "IBIT_CONT": "IBIT",
    "BTC_CONT": "BTC",
    "UCNY_CONT": "UCNY",
    "ED_CONT": "ED",
}


def underlying_family(ticker: str) -> str:
    """Семейство базового актива: `CNYRUBF` и `CNY_CONT` → `CNY`."""
    key = str(ticker).upper()
    if key in FAMILY_BY_TICKER:
        return FAMILY_BY_TICKER[key]
    if key.endswith("_CONT"):
        return key[: -len("_CONT")]
    if key.endswith("RUBF"):
        return key[: -len("RUBF")]
    return key


def deduplicate(
    metrics: Dict[str, float], keep_order: Optional[Iterable[str]] = None
) -> Dict[str, str]:
    """Из каждой семьи оставляет инструмент с лучшей метрикой (обычно OOS Net).

    Возвращает ``{family: ticker}``. При равенстве выигрывает тот, что раньше
    в `keep_order` (детерминированность).
    """
    order = {ticker: index for index, ticker in enumerate(keep_order or [])}
    best: Dict[str, tuple] = {}
    for ticker, value in metrics.items():
        family = underlying_family(ticker)
        candidate = (value, -order.get(ticker, 10_000), ticker)
        if family not in best or candidate > best[family]:
            best[family] = candidate
    return {family: item[2] for family, item in sorted(best.items())}


def load_universe(path: str | Path) -> Dict[str, Any]:
    """Читает `forts_universe.json` (карта фронт-месяцев и серии)."""
    data = Path(path)
    if not data.exists():
        return {}
    try:
        return json.loads(data.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _asset_of(ticker: str, universe: Dict[str, Any]) -> Optional[str]:
    """Код базового актива ISS (`BRM`, `GAZR`, `CNYRUBTOM`) для тикера или его семейства."""
    series = universe.get("series") or {}
    row = series.get(ticker)
    if row:
        return str(row.get("asset") or "")
    for secid, item in series.items():
        if str(secid).upper() == ticker.upper():
            return str(item.get("asset") or "")
    key = str(ticker).upper()
    candidates = {underlying_family(ticker).upper(), key}
    if key.endswith("_CONT"):
        candidates.add(key[: -len("_CONT")])
    for secid, item in series.items():
        asset = str(item.get("asset") or "").upper()
        if asset and asset in candidates:
            return str(item.get("asset") or "")
    return None


def resolve_front_contract(
    ticker: str,
    universe: Dict[str, Any],
    today: Optional[date] = None,
    current: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[str]:
    """`*_CONT` (или перпетуал) → реально торгуемый контракт семейства.

    Порядок: живой список ISS (`current`) → карта `front_by_month` из кэша.
    Возвращает `None`, если резолв невозможен (тогда торговать инструмент нельзя).
    """
    key = str(ticker)
    if not key.upper().endswith("_CONT"):
        return key  # перпетуалы и обычные контракты торгуются как есть
    asset = _asset_of(key, universe)
    # Если инструмента нет в кэше, актив определяем по имени семейства:
    # для `CNY_CONT` это `CNY`, для `BRM_CONT` — `BRM`.
    names = {underlying_family(key).upper(), key.upper()}
    if key.upper().endswith("_CONT"):
        names.add(key[: -len("_CONT")].upper())
    if asset is None and not current:
        return None
    today = today or date.today()

    def _same_asset(value: Any) -> bool:
        text = str(value or "").upper()
        if not text:
            return False
        if asset is not None and text == asset.upper():
            return True
        return text in names

    if current:
        candidates = [row for row in current.values() if _same_asset(row.get("asset"))]
        live = []
        for row in candidates:
            last_trade = str(row.get("last_trade") or "")
            if str(row.get("last_trade")) == "2100-01-01":
                live.append((0, -float(row.get("margin") or 0), str(row.get("secid"))))
                continue
            try:
                expiry = date.fromisoformat(last_trade)
            except ValueError:
                continue
            if expiry >= today:
                live.append((1, expiry.toordinal(), str(row.get("secid"))))
        if live:
            # Сначала «вечный» контракт семейства, иначе ближайший по экспирации.
            live.sort()
            return live[0][2]
    months = universe.get("front_by_month") or {}
    same_family: Dict[str, str] = {}
    for composite, secid in months.items():
        family = composite.split(":")[0]
        if not _same_asset(family):
            continue
        same_family[composite.split(":")[-1]] = str(secid)
    if not same_family:
        return None
    current_month = today.strftime("%Y-%m")
    eligible = [month for month in sorted(same_family) if month <= current_month]
    month = eligible[-1] if eligible else sorted(same_family)[0]
    return str(same_family[month])


def resolve_swing_tickers(
    tickers: Iterable[str],
    universe: Dict[str, Any],
    today: Optional[date] = None,
    current: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Optional[str]]:
    """Карта `preset → живой тикер` для paper/live (бэктест оставляет `*_CONT`)."""
    return {ticker: resolve_front_contract(ticker, universe, today, current) for ticker in tickers}


def needs_resolution(tickers: Iterable[str]) -> List[str]:
    return [ticker for ticker in tickers if str(ticker).upper().endswith("_CONT")]
