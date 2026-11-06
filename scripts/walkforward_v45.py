"""Свип v4.5: сырая цена в комиссии FSM, макро-фильтр EMA_HTF, ресенси-гард, Top-6/8.

    py -3 scripts/walkforward_v45.py --bars 30
    py -3 scripts/walkforward_v45.py --bars 60 --only HTF240
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bot_a.backtest.walkforward import (  # noqa: E402
    WalkForwardSettings,
    build_all_signals,
    run_walkforward,
)
from bot_a.utils.time_provider import MSK  # noqa: E402
from breakout_scan import load_candles, load_meta  # noqa: E402
from multiregime_scan import PERPETUALS, available_continuous, v32_params  # noqa: E402

SPLIT = "2026-07-25"

# (raw_fee, htf) — сигнальные потоки дорогие, строим по одному на группу.
ROUND2: List[Dict[str, Any]] = [
    # Изолируем эффект «сырой цены» при одинаковом min_pf (базовый v4.4 = 1.15).
    {"name": "baseline fair r0.9", "raw": False, "htf": 0, "recent": 0, "top": 12,
     "risk": 0.9, "cap": 0.0, "min_pf": 1.15},
    {"name": "fix fair r0.9", "raw": True, "htf": 0, "recent": 0, "top": 12,
     "risk": 0.9, "cap": 0.0, "min_pf": 1.15},
    # Макро-фильтр: насколько жёстко режем контр-макро.
    {"name": "HTF240 er.30 r0.9", "raw": True, "htf": 240, "recent": 0, "top": 12,
     "risk": 0.9, "cap": 30.0, "er_override": 0.30},
    {"name": "HTF240 slope.30 r0.9", "raw": True, "htf": 240, "recent": 0, "top": 12,
     "risk": 0.9, "cap": 30.0, "slope_tol": 0.30},
    {"name": "HTF180 er.30 r0.9", "raw": True, "htf": 180, "recent": 0, "top": 12,
     "risk": 0.9, "cap": 30.0, "er_override": 0.30},
    # Концентрация + ресенси + риск-тиры.
    {"name": "HTF240 er.30 rec top8 r0.9", "raw": True, "htf": 240, "recent": 15, "top": 8,
     "risk": 0.9, "cap": 30.0, "er_override": 0.30},
    {"name": "HTF240 er.30 rec top8 r1.1", "raw": True, "htf": 240, "recent": 15, "top": 8,
     "risk": 1.1, "cap": 30.0, "er_override": 0.30},
    {"name": "HTF240 er.30 rec top10 r0.9", "raw": True, "htf": 240, "recent": 15, "top": 10,
     "risk": 0.9, "cap": 30.0, "er_override": 0.30},
    {"name": "HTF240 er.30 rec top8 r0.8", "raw": True, "htf": 240, "recent": 15, "top": 8,
     "risk": 0.8, "cap": 30.0, "er_override": 0.30},
]


ROUND3: List[Dict[str, Any]] = [
    # Ресенси-гард и концентрация без макро-фильтра, плюс риск-тиры.
    {"name": "rec top8 r0.9", "raw": True, "htf": 0, "recent": 15, "top": 8,
     "risk": 0.9, "cap": 30.0},
    {"name": "rec top8 r1.1", "raw": True, "htf": 0, "recent": 15, "top": 8,
     "risk": 1.1, "cap": 30.0},
    {"name": "rec top6 r1.2", "raw": True, "htf": 0, "recent": 15, "top": 6,
     "risk": 1.2, "cap": 30.0},
    {"name": "elite pf1.5 rec top6 r1.2", "raw": True, "htf": 0, "recent": 15, "top": 6,
     "risk": 1.2, "cap": 30.0, "min_pf": 1.50},
    {"name": "HTF240 er.40 rec top8 r1.2", "raw": True, "htf": 240, "recent": 15, "top": 8,
     "risk": 1.2, "cap": 30.0},
]

ROUND4: List[Dict[str, Any]] = [
    # 60m: ищем Net > 18 тыс. при DD <= 15 % и 5 положительных месяцах.
    {"name": "60 rec top6 r1.0", "raw": True, "htf": 0, "recent": 15, "top": 6,
     "risk": 1.0, "cap": 30.0},
    {"name": "60 rec top6 r1.1", "raw": True, "htf": 0, "recent": 15, "top": 6,
     "risk": 1.1, "cap": 30.0},
    {"name": "60 rec top8 r1.3", "raw": True, "htf": 0, "recent": 15, "top": 8,
     "risk": 1.3, "cap": 30.0},
    {"name": "60 HTF240 rec top6 r1.0", "raw": True, "htf": 240, "recent": 15, "top": 6,
     "risk": 1.0, "cap": 30.0},
    {"name": "60 HTF240 rec top6 r1.2", "raw": True, "htf": 240, "recent": 15, "top": 6,
     "risk": 1.2, "cap": 30.0},
]

CASES: List[Dict[str, Any]] = ROUND2


def _ms(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=MSK).timestamp() * 1000)


def universe() -> List[str]:
    cont = available_continuous()
    tickers = [t for t in PERPETUALS if any((ROOT / "data_cache").glob(f"iss_forts_{t}_*.json"))]
    tickers += sorted(cont.values())
    return sorted(set(tickers))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Свип v4.5 (макро-фильтр, ресенси, топ-6/8)")
    parser.add_argument("--bars", type=int, default=30)
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--only", default="")
    parser.add_argument("--cases", default="round2", choices=["round2", "round3", "round4"])
    parser.add_argument("--signal-from", default="2026-03-05")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    started = time.time()
    tickers = universe()
    print(f"универсум: {len(tickers)} инструментов, баров {args.bars}", flush=True)
    candles = {
        ticker: load_candles(ticker, args.signal_from, "2026-11-05", args.bars)
        for ticker in tickers
    }
    metas = {ticker: load_meta(ticker) for ticker in tickers}

    source = {"round2": ROUND2, "round3": ROUND3, "round4": ROUND4}[args.cases]
    cases = [case for case in source if (not args.only or args.only in case["name"])]
    groups: Dict[Tuple[bool, int, float, float], List[Dict[str, Any]]] = {}
    for case in cases:
        key = (case["raw"], case["htf"], float(case.get("er_override", 0.40)),
               float(case.get("slope_tol", 0.05)))
        groups.setdefault(key, []).append(case)

    start_ms = _ms("2026-04-25")
    end_ms = _ms("2026-11-05")
    split_ms = _ms(SPLIT)
    weeks = (end_ms - start_ms) / (7 * 86_400_000)
    report: Dict[str, Any] = {"bars": args.bars, "capital": args.capital, "cases": {}}

    print(
        f"\n{'случай':<26} {'сдел':>5} {'сдел/нед':>8} {'OOS':>9} {'IS':>9} {'Full':>9} "
        f"{'PF':>6} {'Sh':>6} {'DD%':>6} {'год%':>7} {'+мес':>5}",
        flush=True,
    )
    for (raw, htf, er_override, slope_tol), group in groups.items():
        params = {ticker: v32_params(args.bars, True) for ticker in tickers}
        params = {
            ticker: params[ticker].overrides(
                fee_on_raw_price=bool(raw),
                htf_ema_bars=int(htf),
                htf_counter_er_override=float(er_override),
                htf_slope_tolerance_atr=float(slope_tol),
            )
            for ticker in tickers
        }
        signals = build_all_signals(metas, params, candles)
        macros = sum(1 for series in signals.values() for s in series if s.er > 0)
        scope = f"raw={raw}, htf={htf}, er>{er_override}, slope_tol={slope_tol}"
        total = sum(len(series) for series in signals.values())
        print(f"\nпотоки ({scope}): сделок {total}, с ER {macros}", flush=True)
        for case in group:
            settings = WalkForwardSettings(
                capital_rub=args.capital,
                risk_pct_per_trade=case["risk"],
                step_days=14,
                lookback_days=45,
                top=int(case["top"]),
                min_pf=float(case.get("min_pf", 1.35)),
                min_trades=3,
                max_concurrent_positions=4,
                conviction_sizing=True,
                free_slot_on_breakeven=True,
                max_total_open_positions=8,
                conviction_leader_boost=1.45,
                throttle_start_pct=6.0,
                warmup_days=45,
                max_position_margin_pct=float(case["cap"]),
                recent_days=int(case["recent"]),
            )
            result = run_walkforward(signals, metas, start_ms, end_ms, settings)
            metrics = result.metrics(args.capital)
            oos = sum(t.net_rub for t in result.trades if t.exit_ms < split_ms)
            is_net = sum(t.net_rub for t in result.trades if t.exit_ms >= split_ms)
            positive = sum(1 for value in metrics["monthly"].values() if value > 0)
            report["cases"][case["name"]] = {
                "params": case,
                "metrics": metrics,
                "oosNet": round(oos, 2),
                "isNet": round(is_net, 2),
                "positiveMonths": positive,
                "rejected": {
                    "slots": result.rejected_by_slots,
                    "risk": result.rejected_by_risk,
                    "margin": result.rejected_by_margin,
                },
                "baskets": result.baskets,
            }
            print(
                f"{case['name']:<26} {metrics['trades']:>5} {metrics['trades'] / weeks:>8.1f} "
                f"{oos:>9.0f} {is_net:>9.0f} {metrics['netRub']:>9.0f} "
                f"{str(metrics['profitFactor']):>6} {str(metrics['sharpe']):>6} "
                f"{metrics['maxDrawdownPct']:>6} {metrics['annualizedPct']:>7.1f} {positive:>5}",
                flush=True,
            )
            months = " ".join(f"{k[5:]}:{v:.0f}" for k, v in metrics["monthly"].items())
            print(f"    месяцы {months}", flush=True)

    out = Path(args.out or f"reports/forts_v45_walkforward_{args.bars}m.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено {out} за {time.time() - started:.0f} c", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
