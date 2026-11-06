"""Сетка параметров стратегии v3.0 на 6-месячной истории FORTS (OOS / IS / Full).

Данные берутся **только из локального кэша** `data_cache/` (см.
`scripts/download_forts_history.py`), поэтому прогон воспроизводим и не зависит от сети.

Пример::

    py -3 scripts/breakout_scan.py
    py -3 scripts/breakout_scan.py --tickers CNYRUBF IMOEXF --jobs 4
    py -3 scripts/breakout_scan.py --periods full --out reports/forts_breakout_full.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot_a.backtest.breakout_backtest import (  # noqa: E402
    BreakoutBacktester,
    BreakoutSettings,
)
from bot_a.backtest.candles import Candle  # noqa: E402
from bot_a.economics.fees import Tariff  # noqa: E402
from bot_a.engine.breakout_fsm import BreakoutParams  # noqa: E402
from bot_a.models.domain import InstrumentMetadata, MarketKind  # noqa: E402
from bot_a.utils.quotation import dec  # noqa: E402
from bot_a.utils.time_provider import MSK  # noqa: E402

DEFAULT_TICKERS = (
    "CNYRUBF",
    "IMOEXF",
    "GLDRUBF",
    "CRZ6",
    "MMZ6",
    "SRZ6",
    "SVZ6",
    "GDZ6",
)
PERIODS = {
    "oos": ("2026-04-25", "2026-07-25"),
    "is": ("2026-07-25", "2026-11-06"),
    "full": ("2026-04-25", "2026-11-06"),
}
CACHE_DIR = ROOT / "data_cache"
META_CACHE = CACHE_DIR / "forts_meta.json"

GRID: Dict[str, Sequence[float]] = {
    "donchian_bars": (20, 30, 45),
    "initial_stop_atr_multiple": (1.0, 1.3),
    "trail_atr_multiple": (1.8, 2.2, 2.8),
    "volume_multiple": (1.3, 1.8),
    "max_squeeze_ratio": (1.35, 1.60, 0.0),
}


def _msk_ms(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=MSK).timestamp() * 1000)


def load_meta(ticker: str) -> InstrumentMetadata:
    """Метаданные из локального кэша (сеть — только при первом обращении)."""
    cache: Dict[str, Dict[str, Any]] = {}
    if META_CACHE.exists():
        cache = json.loads(META_CACHE.read_text(encoding="utf-8"))
    if ticker not in cache:
        from bot_a.backtest.iss_source import fetch_futures_metadata

        meta = fetch_futures_metadata(ticker)
        cache[ticker] = {
            "min_step": str(meta.min_price_increment),
            "step_price": str(meta.min_price_increment_amount),
            "margin": str(meta.initial_margin_buy),
            "lot": meta.lot_size,
            "name": meta.name,
        }
        META_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    row = cache[ticker]
    return InstrumentMetadata(
        instrument_uid=ticker,
        ticker=ticker,
        class_code="SPBFUT",
        market=MarketKind.FUTURES,
        name=str(row.get("name") or ticker),
        lot_size=int(row.get("lot") or 1),
        min_price_increment=dec(row["min_step"]),
        min_price_increment_amount=dec(row["step_price"]),
        initial_margin_buy=dec(row["margin"]),
        initial_margin_sell=dec(row["margin"]),
    )


def load_candles(ticker: str, date_from: str, date_to: str, bar_minutes: int = 1) -> List[Candle]:
    start = _msk_ms(date_from)
    end = _msk_ms(date_to) + 86_400_000 - 1
    candles: List[Candle] = []
    for path in sorted((CACHE_DIR).glob(f"iss_forts_{ticker}_*.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            candle = Candle.from_dict(row)
            if start <= candle.time_ms <= end:
                candles.append(candle)
    candles.sort(key=lambda item: item.time_ms)
    return resample(candles, bar_minutes) if bar_minutes > 1 else candles


def resample(candles: Sequence[Candle], minutes: int) -> List[Candle]:
    """Склеивает 1-минутные свечи в бары по ``minutes`` минут (OHLC + сумма объёма)."""
    if minutes <= 1 or not candles:
        return list(candles)
    width = int(minutes) * 60_000
    result: List[Candle] = []
    bucket: Optional[int] = None
    open_ = high = low = close = None
    volume = 0
    adj = 0.0
    for candle in candles:
        key = (candle.time_ms // width) * width
        if bucket is None or key != bucket:
            if bucket is not None and open_ is not None:
                result.append(
                    Candle(
                        time_ms=bucket,
                        open=open_,
                        high=high,
                        low=low,
                        close=close,
                        volume=volume,
                        adj=adj,
                    )
                )
            bucket = key
            open_ = candle.open
            high = candle.high
            low = candle.low
            close = candle.close
            volume = int(candle.volume)
            adj = candle.adj
            continue
        high = max(high, candle.high)
        low = min(low, candle.low)
        close = candle.close
        volume += int(candle.volume)
    if bucket is not None and open_ is not None:
        result.append(
            Candle(
                time_ms=bucket,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                adj=adj,
            )
        )
    return result


def build_params(combo: Dict[str, float]) -> BreakoutParams:
    """Собирает параметры из комбинации сетки (snake_case или camelCase)."""
    return BreakoutParams.from_mapping(dict(combo))


def _worker(
    task: Tuple[str, Sequence[str], float, float, bool, Dict[str, Sequence[float]], int]
) -> Dict[str, Any]:
    ticker, periods, capital, risk_pct, with_trades, grid, bar_minutes = task
    meta = load_meta(ticker)
    settings = BreakoutSettings(
        capital_rub=capital,
        risk_pct_per_trade=risk_pct,
        tariff=Tariff.PREMIUM,
    )
    out: Dict[str, Any] = {"ticker": ticker, "periods": {}}
    combos = [dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())]
    cache: Dict[str, List[Candle]] = {}
    for period in periods:
        date_from, date_to = PERIODS[period]
        key = f"{date_from}:{date_to}"
        if key not in cache:
            cache[key] = load_candles(ticker, date_from, date_to, bar_minutes)
        candles = cache[key]
        rows: List[Dict[str, Any]] = []
        for combo in combos:
            started = time.time()
            backtester = BreakoutBacktester(meta, build_params(combo), settings)
            result = backtester.run(candles)
            summary = result.summary(capital)
            summary["params"] = combo
            summary["elapsed"] = round(time.time() - started, 2)
            if with_trades:
                summary["tradeList"] = [trade.to_dict() for trade in result.trades]
            rows.append(summary)
        out["periods"][period] = {
            "dateFrom": date_from,
            "dateTo": date_to,
            "candles": len(candles),
            "rows": rows,
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Сетка параметров v3.0 (FORTS breakout)")
    parser.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    parser.add_argument("--periods", nargs="*", default=list(PERIODS.keys()))
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--risk", type=float, default=1.5)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--with-trades", action="store_true")
    parser.add_argument(
        "--grid",
        default=None,
        help='своя сетка JSON, например \'{"donchian_bars":[30],"trail_atr_multiple":[2.0,2.5]}\'',
    )
    parser.add_argument("--grid-file", default=None, help="JSON-файл с сеткой (предпочтительно в PowerShell)")
    parser.add_argument(
        "--bar-minutes",
        type=int,
        default=1,
        help="таймфрейм баров в минутах: 1 (по умолчанию), 5, 15, 60 (история склеивается)",
    )
    parser.add_argument("--out", default="reports/forts_breakout_scan.json")
    args = parser.parse_args(argv)
    grid: Dict[str, Sequence[float]] = dict(GRID)
    if args.grid_file:
        grid.update(json.loads(Path(args.grid_file).read_text(encoding="utf-8")))
    if args.grid:
        grid.update(json.loads(args.grid))
    jobs = args.jobs or max(1, min(len(args.tickers), (mp.cpu_count() or 2)))
    tasks = [
        (
            ticker,
            tuple(args.periods),
            args.capital,
            args.risk,
            bool(args.with_trades),
            grid,
            int(args.bar_minutes),
        )
        for ticker in args.tickers
        if ticker in {name for name in args.tickers}
    ]
    payload: Dict[str, Any] = {
        "generated": datetime.now(tz=MSK).isoformat(timespec="seconds"),
        "capital": args.capital,
        "riskPct": args.risk,
        "tariff": Tariff.PREMIUM.value,
        "grid": {key: list(values) for key, values in grid.items()},
        "barMinutes": int(args.bar_minutes),
        "results": {},
    }
    started = time.time()
    with mp.Pool(processes=jobs) as pool:
        for index, chunk in enumerate(pool.imap_unordered(_worker, tasks), start=1):
            payload["results"][chunk["ticker"]] = chunk["periods"]
            print(
                f"[{index}/{len(tasks)}] {chunk['ticker']} готов за {time.time() - started:.1f} c",
                flush=True,
            )
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"сохранено: {out_path} ({time.time() - started:.1f} c)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
