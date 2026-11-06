"""Свип v3.1: таймфрейм × режим входа × перенос через ночь на FX/товарной корзине.

Проверяет три гипотезы §8 отчёта v3.0:

1. **старший таймфрейм** (`15 / 30 / 60` минут) снижает долю комиссии в риске сделки;
2. **перенос позиции через ночь** (FORTS не берёт маржинальный процент) сохраняет тренды,
   которые интрадей-правило `18:35` срезало;
3. **вход по касанию уровня** (`entryMode = stop_level`) убирает «догон» бара пробоя.

Параметры **единые для всех инструментов** (без подгонки по тикеру). Результат — JSON
и таблица по периодам OOS / IS / Full.

Пример::

    py -3 scripts/breakout_swing_scan.py
    py -3 scripts/breakout_swing_scan.py --tickers CNYRUBF GLDRUBF --timeframes 60
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
sys.path.insert(0, str(ROOT / "scripts"))

from bot_a.backtest.breakout_backtest import (  # noqa: E402
    BreakoutBacktester,
    BreakoutSettings,
    aggregate,
)
from bot_a.economics.fees import Tariff  # noqa: E402
from bot_a.engine.breakout_fsm import BreakoutParams  # noqa: E402
from bot_a.utils.time_provider import MSK  # noqa: E402
from breakout_scan import load_candles, load_meta  # noqa: E402

# FX + товарная корзина: макро-тренды живут днями (отчёт v3.0 §8).
DEFAULT_TICKERS = ("CNYRUBF", "GLDRUBF", "NGZ6", "SVZ6", "USDRUBF")
PERIODS = {
    "oos": ("2026-04-25", "2026-07-25"),
    "is": ("2026-07-25", "2026-11-06"),
    "full": ("2026-04-25", "2026-11-06"),
}
SWING_WINDOWS = "10:15-13:45,14:10-18:35,19:10-22:30"
INTRADAY_WINDOWS = "10:15-13:45,14:10-18:20"

# Варианты выхода: (метка, переопределения)
EXIT_VARIANTS: Dict[str, Dict[str, Any]] = {
    "base": {},
    "wide": {
        "initial_stop_atr_multiple": 2.5,
        "trail_atr_multiple": 3.0,
        "trail_trigger_atr": 2.5,
        "breakeven_trigger_atr": 3.0,
    },
    "tight": {
        "initial_stop_atr_multiple": 1.5,
        "trail_atr_multiple": 2.5,
        "trail_trigger_atr": 1.5,
        "breakeven_trigger_atr": 2.0,
    },
    # «Держим тренд»: без тейка, широкий трейл — для многосуточных свингов.
    "run": {
        "take_profit_atr": 0.0,
        "trail_atr_multiple": 4.0,
        "trail_trigger_atr": 2.5,
        "breakeven_trigger_atr": 3.0,
    },
    "run2": {
        "take_profit_atr": 0.0,
        "trail_atr_multiple": 3.0,
        "trail_trigger_atr": 2.0,
        "breakeven_trigger_atr": 2.5,
        "initial_stop_atr_multiple": 2.5,
    },
}

BASE_PARAMS: Dict[str, Any] = {
    "donchian_bars": 20,
    "atr_bars": 15,
    "squeeze_median_bars": 120,
    "max_squeeze_ratio": 1.6,
    "ema_fast_bars": 20,
    "ema_slow_bars": 90,
    "volume_lookback": 20,
    "volume_multiple": 1.5,
    "close_in_range": 0.3,
    "min_atr_to_fee_ratio": 2.5,
    "obi_long_threshold": 0.2,
    "obi_short_threshold": -0.2,
    "breakout_buffer_steps": 1,
    "max_entry_extension_atr": 0.35,
    "initial_stop_atr_multiple": 2.0,
    "min_stop_steps": 6,
    "failed_breakout_minutes": 0,
    "failed_breakout_bars": 2,
    "failed_breakout_profit_atr": 0.6,
    "breakeven_trigger_atr": 2.5,
    "breakeven_fee_multiple": 1.5,
    "breakeven_lock_atr": 1.0,
    "trail_trigger_atr": 2.0,
    "trail_atr_multiple": 3.0,
    "take_profit_atr": 5.0,
    "max_consecutive_stops": 2,
    "cooldown_minutes": 45,
    "slippage_steps": 1,
    "loser_close_bars": 2,
    "clearing_close": "18:40,23:40",
}


def build_params(
    bar_minutes: int, entry_mode: str, overnight: bool, variant: str
) -> BreakoutParams:
    overrides = dict(BASE_PARAMS)
    overrides.update(EXIT_VARIANTS[variant])
    overrides["entry_mode"] = entry_mode
    overrides["allow_overnight_hold"] = overnight
    overrides["close_losers_before_clearing"] = overnight
    overrides["session_windows"] = SWING_WINDOWS if overnight else INTRADAY_WINDOWS
    overrides["trade_until"] = "18:35"
    return BreakoutParams.from_mapping(overrides)


def _worker(task: Tuple[str, Sequence[int], Sequence[str], float, float]) -> Dict[str, Any]:
    ticker, timeframes, variants, capital, risk_pct = task
    meta = load_meta(ticker)
    settings = BreakoutSettings(
        capital_rub=capital, risk_pct_per_trade=risk_pct, tariff=Tariff.PREMIUM
    )
    out: Dict[str, Any] = {"ticker": ticker, "series": []}
    for bar_minutes in timeframes:
        candles_by_period = {
            period: load_candles(ticker, *PERIODS[period], bar_minutes) for period in PERIODS
        }
        for entry_mode, overnight, variant in itertools.product(
            ("bar_close", "stop_level"), (False, True), variants
        ):
            params = build_params(bar_minutes, entry_mode, overnight, variant)
            row: Dict[str, Any] = {
                "barMinutes": bar_minutes,
                "entryMode": entry_mode,
                "overnight": overnight,
                "variant": variant,
                "params": params.to_dict(),
            }
            for period in PERIODS:
                candles = candles_by_period[period]
                if not candles:
                    row[period] = None
                    continue
                result = BreakoutBacktester(meta, params, settings).run(candles)
                summary = result.summary(capital)
                summary.pop("counters", None)
                row[period] = summary
            out["series"].append(row)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Свип v3.1 (таймфрейм × вход × перенос)")
    parser.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    parser.add_argument("--timeframes", nargs="*", type=int, default=[15, 30, 60])
    parser.add_argument("--variants", nargs="*", default=list(EXIT_VARIANTS))
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--risk", type=float, default=1.5)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--out", default="reports/forts_swing_scan.json")
    args = parser.parse_args(argv)
    jobs = args.jobs or max(1, min(len(args.tickers), mp.cpu_count() or 2))
    tasks = [
        (ticker, tuple(args.timeframes), tuple(args.variants), args.capital, args.risk)
        for ticker in args.tickers
    ]
    started = time.time()
    payload: Dict[str, Any] = {
        "generated": datetime.now(tz=MSK).isoformat(timespec="seconds"),
        "capital": args.capital,
        "riskPct": args.risk,
        "tariff": Tariff.PREMIUM.value,
        "baseParams": BASE_PARAMS,
        "variants": EXIT_VARIANTS,
        "periods": {name: list(bounds) for name, bounds in PERIODS.items()},
        "results": {},
    }
    with mp.Pool(processes=jobs) as pool:
        for index, chunk in enumerate(pool.imap_unordered(_worker, tasks), start=1):
            payload["results"][chunk["ticker"]] = chunk["series"]
            print(
                f"[{index}/{len(tasks)}] {chunk['ticker']}: серий {len(chunk['series'])} "
                f"за {time.time() - started:.1f} c",
                flush=True,
            )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"сохранено: {out} ({time.time() - started:.1f} c)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
