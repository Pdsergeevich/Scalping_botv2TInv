"""Автообнаружение ликвидного универсума FORTS через MOEX ISS + склейка continuous-серий.

Скрипт не «угадывает» тикеры: он спрашивает у ISS, что реально торговалось в окне
`25.04.2026 – 05.11.2026`, и собирает непрерывные серии по семействам (`ASSETCODE`).

Как это работает:

1. **Текущий список** (`/engines/futures/markets/forts/securities.json`, один запрос) —
   даёт `MINSTEP`, `STEPPRICE`, `INITIALMARGIN`, `LOTVOLUME`, `LASTTRADEDATE` по живым контрактам.
2. **Месячные листинги** (`/history/engines/futures/markets/forts/securities.json?date=...`) —
   дают контракты, которые уже истекли (их нет в текущем списке), вместе с дневным `VALUE`.
   По выборке из ~8 дат определяем ликвидность и «фронтральность» каждого контракта.
3. **Выбор серий**: перпетуалы (`LASTTRADEDATE = 2100-01-01`) + контракты, которые хотя бы в
   одном месяце окна были самым оборотистым контрактом своего семейства (то есть были фронтом).
4. **Загрузка 1-мин свечей** в формате кэша `iss_forts_{ticker}_{yyyymmdd}.json`
   (параллельно, с повторным использованием `IssCandleSource._fetch_day`).
5. **Склейка continuous** (`{ASSET}_CONT`): для каждого торгового дня берём серию с максимальным
   дневным объёмом среди серий этого семейства — это и есть настоящий фронт-месяц без гэпов
   от экспирации. Свечи пишутся в тот же кэш, метаданные — в `forts_meta.json`, поэтому
   `breakout_scan.load_candles` / `load_meta` работают без изменений.

Примеры::

    py -3 scripts/discover_forts_universe.py --stage discover
    py -3 scripts/discover_forts_universe.py --stage download --jobs 12
    py -3 scripts/discover_forts_universe.py --stage stitch
    py -3 scripts/discover_forts_universe.py --stage all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot_a.backtest.iss_source import ISS_BASE, IssCandleSource, _get_json  # noqa: E402
from bot_a.models.domain import InstrumentMetadata, MarketKind  # noqa: E402
from bot_a.utils.quotation import dec  # noqa: E402
from bot_a.utils.time_provider import MSK  # noqa: E402

CACHE_DIR = ROOT / "data_cache"
META_CACHE = CACHE_DIR / "forts_meta.json"
UNIVERSE_JSON = CACHE_DIR / "forts_universe.json"
ROLLS_CACHE = CACHE_DIR / "forts_rolls.json"
DATE_FROM = "2026-04-25"
DATE_TO = "2026-11-05"
MAX_MARGIN = 15_000.0
MIN_FAMILY_VALUE = 5_000_000_000.0  # ₽ оборота семейства за срезы
MIN_SERIES_VALUE = 50_000_000.0  # ₽ дневного оборота серии во фронт-месяце
SAMPLE_DAYS = ("05", "19")  # два среза на месяц: фронт успевает прокрутиться
# Перпетуалы держим только по «российским» базовым активам: у ISS есть сотни вечных
# контрактов на иностранные акции и крипту (TSLAF, QQQF, XRPUSDF…), они не в скоупе.
PERPETUAL_SCOPE = {
    "CNYRUBTOM",
    "USDRUBTOM",
    "EURRUBTOM",
    "GLDRUBTOM",
    "SLVRUBTOM",
    "IMOEX",
    "SBERF",
    "GAZPF",
}


def _ms(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=MSK).timestamp() * 1000)


def _trading_days() -> List[date]:
    """Будни окна (праздники не важны: ISS просто вернёт меньше строк)."""
    start = date.fromisoformat(DATE_FROM)
    end = date.fromisoformat(DATE_TO)
    days = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def fetch_current() -> Dict[str, Dict[str, Any]]:
    """Живые контракты с шагом, стоимостью шага и ГО (один запрос к ISS)."""
    payload = _get_json(
        f"{ISS_BASE}/engines/futures/markets/forts/securities.json?iss.meta=off"
    )["securities"]
    index = {name: position for position, name in enumerate(payload["columns"])}
    out: Dict[str, Dict[str, Any]] = {}
    for row in payload["data"]:
        secid = str(row[index["SECID"]])
        out[secid] = {
            "secid": secid,
            "asset": str(row[index["ASSETCODE"]] or secid),
            "shortname": str(row[index["SHORTNAME"]] or secid),
            "min_step": str(row[index["MINSTEP"]] or "0.01"),
            "step_price": str(row[index["STEPPRICE"]] or "0"),
            "margin": float(row[index["INITIALMARGIN"]] or 0),
            "lot": int(row[index["LOTVOLUME"]] or 1),
            "last_trade": str(row[index["LASTTRADEDATE"]] or ""),
            "price": float(row[index["PREVPRICE"]] or row[index["PREVSETTLEPRICE"]] or 0),
        }
    return out


def fetch_listing(day: date) -> List[Dict[str, Any]]:
    """Все контракты, торговавшиеся в этот день (постранично, включая истёкшие)."""
    rows: List[Dict[str, Any]] = []
    start = 0
    while True:
        url = (
            f"{ISS_BASE}/history/engines/futures/markets/forts/securities.json"
            f"?date={day.isoformat()}&start={start}&iss.meta=off"
        )
        payload = None
        for attempt in range(4):
            try:
                payload = _get_json(url)["history"]
                break
            except Exception:  # noqa: BLE001 — ISS троттлит частые запросы
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        if payload is None:
            break
        columns = payload["columns"]
        data = payload["data"]
        if not data:
            break
        index = {name: position for position, name in enumerate(columns)}
        for row in data:
            if str(row[index["BOARDID"]]) != "RFUD":
                continue
            rows.append(
                {
                    "secid": str(row[index["SECID"]]),
                    "asset": str(row[index["ASSETCODE"]] or ""),
                    "value": float(row[index["VALUE"]] or 0),
                    "volume": float(row[index["VOLUME"]] or 0),
                    "close": float(row[index["CLOSE"]] or 0),
                    "day": day.isoformat(),
                }
            )
        start += len(data)
        if len(data) < 100:
            break
    return rows


def sample_listings(max_days: int = 16) -> List[Dict[str, Any]]:
    """Срезы листинга: 2 даты в месяц (15-е и 26-е число или ближайший будний)."""
    days = _trading_days()
    picks: List[date] = []
    seen_months = set()
    for day in days:
        key = (day.year, day.month)
        if key not in seen_months and day.day >= 14:
            picks.append(day)
            seen_months.add(key)
        if len(picks) >= max_days:
            break
    rows: List[Dict[str, Any]] = []
    for day in picks:
        chunk = fetch_listing(day)
        rows.extend(chunk)
        print(f"  листинг {day}: контрактов {len(chunk)}", flush=True)
    return rows


def build_universe(
    current: Dict[str, Dict[str, Any]], listings: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Выбор серий: перпетуалы + бывшие фронт-месяцы ликвидных семейств."""
    per_series: Dict[str, Dict[str, Any]] = {}
    for row in listings:
        entry = per_series.setdefault(
            row["secid"],
            {"secid": row["secid"], "asset": row["asset"], "value": 0.0, "closes": [], "days": 0, "months": []},
        )
        entry["value"] += row["value"]
        entry["days"] += 1
        if row["day"][:7] not in entry["months"]:
            entry["months"].append(row["day"][:7])
        if row["close"] > 0:
            entry["closes"].append(row["close"])
    # Оборот семейств.
    family_value: Dict[str, float] = defaultdict(float)
    for entry in per_series.values():
        family_value[entry["asset"]] += entry["value"]

    # Лучший контракт семейства в разрезе месяца (фронт-месяц).
    front_by_month: Dict[Tuple[str, str], Tuple[str, float]] = {}
    for row in listings:
        key = (row["asset"], row["day"][:7])
        best = front_by_month.get(key)
        if best is None or row["value"] > best[1]:
            front_by_month[key] = (row["secid"], row["value"])

    selected: Dict[str, Dict[str, Any]] = {}
    for secid, info in current.items():
        if str(info["last_trade"]) != "2100-01-01":
            continue
        if info["asset"] not in PERPETUAL_SCOPE:
            continue
        selected[secid] = {"secid": secid, "asset": info["asset"], "reason": "perpetual"}
    for (asset, month), (secid, value) in front_by_month.items():
        if family_value[asset] < MIN_FAMILY_VALUE:
            continue
        if value < MIN_SERIES_VALUE:
            continue
        # ISS отдаёт «склеенные» псевдо-тикеры (ANU6ANZ6, EDM6EDU6) — это артефакт,
        # а не контракт: у настоящих серий код ≤ 5 символов.
        if len(secid) > 5 and not secid.endswith("F"):
            continue
        selected.setdefault(secid, {"secid": secid, "asset": asset, "reason": f"front {month}"})

    # Метаданные: спецификация и ГО. Для истёкших — прокси от живого контракта семейства
    # с масштабированием ГО по отношению цен (ГО ≈ доля нотионала).
    spec: Dict[str, Dict[str, Any]] = {}
    for info in current.values():
        family = info["asset"]
        if info["margin"] <= 0 or float(info["min_step"]) <= 0:
            continue
        if family not in spec or info["margin"] < spec[family]["margin"]:
            spec[family] = dict(info)
    meta: Dict[str, Dict[str, Any]] = {}
    for secid, info in selected.items():
        asset = info["asset"]
        base = current.get(secid) or spec.get(asset)
        if base is None:
            continue
        closes = per_series.get(secid, {}).get("closes") or []
        price = float(sum(closes) / len(closes)) if closes else float(base["price"] or 0)
        ref = float(base["price"] or 0) or price or 1.0
        scale = (price / ref) if price > 0 and ref > 0 else 1.0
        margin = float(base["margin"]) * (scale if base is not current.get(secid) else 1.0)
        info.update(
            {
                "min_step": base["min_step"],
                "step_price": base["step_price"],
                "margin": round(margin, 2),
                "lot": base["lot"],
                "price": price,
                "shortname": base.get("shortname") or secid,
                "turnover": round(per_series.get(secid, {}).get("value", 0.0), 0),
                "months": per_series.get(secid, {}).get("months", []),
            }
        )
        if info["margin"] > MAX_MARGIN:
            info["reason"] += " (ГО > лимита)"
        meta[secid] = info
    return {
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
        "families": sorted(family_value.items(), key=lambda item: -item[1]),
        "series": meta,
        "front_by_month": {f"{asset}:{month}": secid for (asset, month), (secid, _) in front_by_month.items()},
    }


def _meta_for(ticker: str, row: Dict[str, Any]) -> InstrumentMetadata:
    return InstrumentMetadata(
        instrument_uid=ticker,
        ticker=ticker,
        class_code="SPBFUT",
        market=MarketKind.FUTURES,
        name=str(row.get("shortname") or ticker),
        lot_size=int(row.get("lot") or 1),
        min_price_increment=dec(row.get("min_step") or "0.01"),
        min_price_increment_amount=dec(row.get("step_price") or "0"),
        initial_margin_buy=dec(str(row.get("margin") or 0)),
        initial_margin_sell=dec(str(row.get("margin") or 0)),
    )


def download_series(series: Dict[str, Dict[str, Any]], jobs: int) -> None:
    """Качает 1-мин свечи по всем выбранным сериям (параллельно, с кэшем).

    Окно каждой серии ограничено месяцами, в которых она реально торговалась
    (по месячным листингам): у неликвидных дальних контрактов нечего качать.
    """
    source = IssCandleSource(CACHE_DIR)
    days = _trading_days()
    window_start = date.fromisoformat(DATE_FROM)
    window_end = date.fromisoformat(DATE_TO)
    tasks: List[Tuple[str, InstrumentMetadata, date]] = []
    for secid, row in sorted(series.items()):
        meta = _meta_for(secid, row)
        months = sorted(row.get("months") or [])
        if months and "perpetual" not in row["reason"]:
            first = date.fromisoformat(f"{months[0]}-01")
            last_month = date.fromisoformat(f"{months[-1]}-01")
            last = (last_month + timedelta(days=31)).replace(day=1) - timedelta(days=1)
            start = max(window_start, first - timedelta(days=12))
            end = min(window_end, last + timedelta(days=6))
        else:
            start, end = window_start, window_end
        for day in days:
            if not (start <= day <= end):
                continue
            path = source._cache_path(meta, day)
            if not path.exists():
                tasks.append((secid, meta, day))
    print(f"свечей к загрузке: {len(tasks)} (серий {len(series)}, дней {len(days)})", flush=True)

    done = {"count": 0, "empty": 0}
    started = time.time()

    def work(task: Tuple[str, InstrumentMetadata, date]) -> Tuple[str, date, int]:
        secid, meta, day = task
        try:
            rows = source._fetch_day(meta, day)
        except Exception as exc:  # noqa: BLE001 — сеть не должна ронять прогон
            return secid, day, -1
        (source._cache_path(meta, day)).write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
        return secid, day, len(rows)

    with ThreadPoolExecutor(max_workers=max(int(jobs), 1)) as pool:
        for secid, day, count in pool.map(work, tasks):
            done["count"] += 1
            if count <= 0:
                done["empty"] += 1
            if done["count"] % 200 == 0:
                speed = done["count"] / max(time.time() - started, 1e-9)
                print(
                    f"  {done['count']}/{len(tasks)} ({speed:.1f} дн/с), пусто {done['empty']}",
                    flush=True,
                )
    print(
        f"готово: {done['count']} дней за {time.time() - started:.1f} c, пусто {done['empty']}",
        flush=True,
    )


def _load_day(ticker: str, day: str) -> List[Dict[str, Any]]:
    path = CACHE_DIR / f"iss_forts_{ticker}_{day}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")) or []
    except (OSError, json.JSONDecodeError):
        return []


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _roll_delta(
    old_ticker: str,
    new_ticker: str,
    day: str,
    prev_day: str,
    prev_rows: List[Dict[str, Any]],
) -> Tuple[float, str]:
    """Спред ролла ΔP = P_новый − P_старый на дне переключения.

    Основной способ — медиана разниц по **общим минутам** дня ролла (обе серии
    ещё торгуются): так в ΔP не попадает обычный ночной гэп. Если пересечения
    нет (старый контракт уже неликвид или данных мало) — оценка по стыку
    `open нового` − `close старого` предыдущего дня.
    """
    old_rows = _load_day(old_ticker, day)
    new_rows = _load_day(new_ticker, day)
    if old_rows and new_rows:
        old_close = {int(row["time"]): float(row["close"]) for row in old_rows}
        diffs = [
            float(row["close"]) - old_close[int(row["time"])]
            for row in new_rows
            if int(row["time"]) in old_close
        ]
        if len(diffs) >= 10:
            return _median(diffs), "overlap"
    last_old = float(prev_rows[-1]["close"]) if prev_rows else 0.0
    first_new = float(new_rows[0]["open"]) if new_rows else last_old
    if not last_old:
        return 0.0, "none"
    return first_new - last_old, "edge"


def stitch(universe: Dict[str, Any]) -> Dict[str, str]:
    """Склейка continuous по семействам с back-adjustment на роллах.

    Каждый день берётся серия с максимальным дневным объёмом (реальный фронт).
    На дне переключения контракта считается спред ролла ΔP = P_новый − P_старый
    и **вся более старая история сдвигается на накопленную сумму ΔP** (difference /
    back-adjustment, как в непрерывных сериях вендоров): после этого в ряду нет
    фантомных скачков `+357 пт` (GAZR) и `−575 пт` (VTBR), которые ломали
    `EMA20/EMA90`, `ATR`, `ER_20` и PnL открытых сделок.

    Исходные цены восстановимы: в каждой свече пишется поле ``adj`` (на сколько
    пунктов её сдвинули), поэтому комиссия и ГО считаются от ``price - adj``.
    Роллы складываются в ``data_cache/forts_rolls.json``.
    """
    series = universe["series"]
    by_family: Dict[str, List[str]] = defaultdict(list)
    for secid, row in series.items():
        if "perpetual" in row["reason"]:
            continue
        if row["margin"] > MAX_MARGIN:
            continue
        by_family[row["asset"]].append(secid)

    # Лучшая серия семейства на каждый день: (asset, day) -> (volume, ticker, rows)
    daily: Dict[Tuple[str, str], Tuple[float, str, List[Dict[str, Any]]]] = {}
    for path in CACHE_DIR.glob("iss_forts_*.json"):
        parts = path.stem.split("_")
        if len(parts) != 4 or parts[2].endswith("_CONT"):
            continue
        ticker = parts[2]
        family = series.get(ticker, {}).get("asset")
        if family is None or ticker not in by_family.get(family, []):
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not rows:
            continue
        volume = sum(float(item.get("volume") or 0) * float(item.get("close") or 0) for item in rows)
        key = (family, parts[3])
        best = daily.get(key)
        if best is None or volume > best[0]:
            daily[key] = (volume, ticker, rows)

    summary: Dict[str, str] = {}
    rolls_out: Dict[str, List[Dict[str, Any]]] = {}
    for family in sorted(by_family):
        picked = sorted(
            (day, ticker, rows)
            for (fam, day), (_volume, ticker, rows) in daily.items()
            if fam == family
        )
        if not picked:
            continue
        ticker = f"{family}_CONT"

        # 1) Роллы: где меняется активный контракт и какой спред между ними.
        delta_at: Dict[int, float] = {}
        rolls: List[Dict[str, Any]] = []
        for index in range(1, len(picked)):
            day, active, _rows = picked[index]
            prev_day, prev_ticker, prev_rows = picked[index - 1]
            if active == prev_ticker:
                continue
            delta, mode = _roll_delta(prev_ticker, active, day, prev_day, prev_rows)
            delta_at[index] = delta
            rolls.append(
                {
                    "day": day,
                    "from": prev_ticker,
                    "to": active,
                    "delta": round(delta, 4),
                    "mode": mode,
                }
            )

        # 2) Back-adjustment: самая свежая сегментация остаётся без сдвига,
        #    каждая более старая сдвигается на накопленную сумму ΔP.
        adj = [0.0] * len(picked)
        for index in range(len(picked) - 1, 0, -1):
            adj[index - 1] = adj[index] + delta_at.get(index, 0.0)

        merged_by_time: Dict[int, Dict[str, Any]] = {}
        for index, (_day, _active, rows) in enumerate(picked):
            offset = adj[index]
            for item in rows:
                record = dict(item)
                if offset:
                    for field in ("open", "high", "low", "close"):
                        if record.get(field) is not None:
                            record[field] = str(Decimal(str(record[field])) + Decimal(str(round(offset, 6))))
                record["adj"] = round(offset, 6)
                merged_by_time[int(record["time"])] = record
        candles = [merged_by_time[key] for key in sorted(merged_by_time)]

        # 3) Раскладываем по дням в тот же формат кэша, что читает load_candles.
        per_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in candles:
            stamp = datetime.fromtimestamp(item["time"] / 1000, tz=MSK).strftime("%Y%m%d")
            per_day[stamp].append(item)
        for old in CACHE_DIR.glob(f"iss_forts_{ticker}_*.json"):
            old.unlink()
        for day, rows in per_day.items():
            path = CACHE_DIR / f"iss_forts_{ticker}_{day}.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        days = len(per_day)
        rolls_out[ticker] = rolls
        worst = max(rolls, key=lambda item: abs(item["delta"]), default=None)
        detail = (
            f"; роллов {len(rolls)}, макс |ΔP| {float(worst['delta']):+.1f} пт ({worst['day']} {worst['from']}→{worst['to']})"
            if worst
            else "; роллов нет"
        )
        print(f"{ticker:<10} дней {days:>3}  свечей {len(candles):>6}{detail}", flush=True)
        summary[family] = ticker

    ROLLS_CACHE.write_text(json.dumps(rolls_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"роллы записаны: {ROLLS_CACHE.name}", flush=True)
    return summary


def write_meta(universe: Dict[str, Any], stitched: Dict[str, str]) -> None:
    """Дописывает выбранные серии и continuous-тикеры в `forts_meta.json`."""
    cache: Dict[str, Dict[str, Any]] = {}
    if META_CACHE.exists():
        cache = json.loads(META_CACHE.read_text(encoding="utf-8"))
    for secid, row in universe["series"].items():
        cache[secid] = {
            "min_step": row["min_step"],
            "step_price": row["step_price"],
            "margin": str(row["margin"]),
            "lot": row["lot"],
            "name": row.get("shortname") or secid,
        }
    for family, ticker in stitched.items():
        members = [
            row for row in universe["series"].values() if row["asset"] == family and "perpetual" not in row["reason"]
        ]
        if not members:
            continue
        base = max(members, key=lambda row: row.get("turnover") or 0)
        cache[ticker] = {
            "min_step": base["min_step"],
            "step_price": base["step_price"],
            "margin": str(round(base["margin"] * 1.1, 2)),
            "lot": base["lot"],
            "name": f"{family} continuous",
        }
    META_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"метаданные обновлены: {len(cache)} инструментов", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    global MAX_MARGIN, DATE_FROM, DATE_TO
    parser = argparse.ArgumentParser(description="Универсум FORTS: обнаружение, загрузка, склейка")
    parser.add_argument("--stage", default="all", choices=["discover", "download", "stitch", "all"])
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--max-margin", type=float, default=MAX_MARGIN)
    parser.add_argument("--max-series", type=int, default=60)
    parser.add_argument("--refresh", action="store_true", help="пересобрать универсум с ISS")
    parser.add_argument("--from", dest="date_from", default=DATE_FROM,
                        help="начало окна докачки (v4.3: 2026-03-05 для прогрева walk-forward)")
    parser.add_argument("--to", dest="date_to", default=DATE_TO, help="конец окна докачки")
    args = parser.parse_args(argv)

    MAX_MARGIN = float(args.max_margin)
    # Прогрев (v4.3): можно задать своё окно — качаем только свечи, универсум из кэша.
    DATE_FROM = str(args.date_from)
    DATE_TO = str(args.date_to)
    started = time.time()
    current = fetch_current()
    print(f"живых контрактов FORTS: {len(current)}", flush=True)

    if UNIVERSE_JSON.exists() and args.stage in ("download", "stitch", "all") and not args.refresh:
        universe = json.loads(UNIVERSE_JSON.read_text(encoding="utf-8"))
    else:
        listings = sample_listings()
        universe = build_universe(current, listings)
        UNIVERSE_JSON.write_text(json.dumps(universe, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nсемейства по обороту (₽ за срезы):")
        for asset, value in universe["families"][:20]:
            print(f"  {asset:<10} {value:>14,.0f}")
        print("\nвыбранные серии:")
        for secid, row in sorted(universe["series"].items(), key=lambda item: item[1]["asset"]):
            print(
                f"  {secid:<10} {row['asset']:<8} ГО {row['margin']:>9,.2f} ₽  "
                f"шаг {row['min_step']}×{row['step_price']}  {row['reason']}"
            )
    if args.stage == "discover":
        return 0

    tradable = {
        secid: row
        for secid, row in universe["series"].items()
        if row["margin"] <= MAX_MARGIN
        and ("perpetual" in row["reason"] or row.get("turnover", 0) >= MIN_SERIES_VALUE)
    }
    tradable = dict(list(sorted(tradable.items(), key=lambda item: -(item[1].get("turnover") or 0)))[: args.max_series])
    print(f"\nк загрузке серий: {len(tradable)}", flush=True)
    if args.stage in ("download", "all"):
        download_series(tradable, args.jobs)
    stitched = stitch(universe)
    write_meta(universe, stitched)
    print(f"\nвсего {time.time() - started:.0f} c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
