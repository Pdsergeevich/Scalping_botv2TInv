"""Свип v4.4: back-adjustment роллов, анти-застой, лимит ГО на позицию (30m/15m).

Кейсы сравниваются на одинаковых сигнальных потоках (кроме вариантов застоя:
там `stallBreakevenAtr`/`stallBars` меняют сами сделки).

    py -3 scripts/walkforward_v44.py --bars 30
    py -3 scripts/walkforward_v44.py --bars 30 --only stall
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

PERIOD_SPLIT = "2026-07-25"


def _ms(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=MSK).timestamp() * 1000)


def universe() -> List[str]:
    cont = available_continuous()
    tickers = [t for t in PERPETUALS if any((ROOT / "data_cache").glob(f"iss_forts_{t}_*.json"))]
    tickers += sorted(cont.values())
    return sorted(set(tickers))


CASES: List[Dict[str, Any]] = [
    # Эталон v4.3: чужие индексы разрешены, застоя нет, лимита ГО на позицию нет.
    {"name": "v43 baseline r0.9", "risk": 0.9, "exclude": (), "stall": None, "cap": 0.0},
    # v4.4: back-adjustment роллов + барьеры переключения + чужие индексы исключены.
    {"name": "v44 rolladj r0.9", "risk": 0.9, "stall": None, "cap": 0.0},
    {"name": "v44 rolladj r0.7", "risk": 0.7, "stall": None, "cap": 0.0},
    {"name": "v44 rolladj r1.1", "risk": 1.1, "stall": None, "cap": 0.0},
    {"name": "v44 rolladj cap35 r0.9", "risk": 0.9, "stall": None, "cap": 35.0},
    {"name": "v44 rolladj cap50 r0.9", "risk": 0.9, "stall": None, "cap": 50.0},
    # Анти-застой: разные пороги/фиксация прибыли.
    {"name": "stall1.5x4 lock0 r0.9", "risk": 0.9, "stall": (1.5, 4, 0.0), "cap": 35.0},
    {"name": "stall2.0x6 lock1.0 r0.9", "risk": 0.9, "stall": (2.0, 6, 1.0), "cap": 35.0},
    {"name": "stall2.5x8 lock1.5 r0.9", "risk": 0.9, "stall": (2.5, 8, 1.5), "cap": 35.0},
]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Свип v4.4 (роллы, застой, лимит ГО)")
    parser.add_argument("--bars", type=int, default=30)
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--only", default="")
    parser.add_argument("--signal-from", default="2026-03-05")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    started = time.time()
    tickers = universe()
    print(f"универсум: {len(tickers)} инструментов", flush=True)
    candles = {
        ticker: load_candles(ticker, args.signal_from, "2026-11-05", args.bars)
        for ticker in tickers
    }
    metas = {ticker: load_meta(ticker) for ticker in tickers}

    cases = [case for case in CASES if (not args.only or args.only in case["name"])]
    stall_groups: Dict[Optional[Tuple[float, int, float]], List[Dict[str, Any]]] = {}
    for case in cases:
        stall_groups.setdefault(case["stall"], []).append(case)

    start_ms = _ms("2026-04-25")
    end_ms = _ms("2026-11-05")
    split_ms = _ms(PERIOD_SPLIT)
    weeks = (end_ms - start_ms) / (7 * 86_400_000)

    report: Dict[str, Any] = {"bars": args.bars, "capital": args.capital, "cases": {}}
    print(
        f"\n{'случай':<30} {'сдел':>5} {'сдел/нед':>8} {'OOS':>9} {'IS':>9} {'Full':>9} "
        f"{'PF':>6} {'Sh':>6} {'DD%':>6} {'год%':>7} {'сбор':>7}",
        flush=True,
    )
    for stall, group in stall_groups.items():
        params = {ticker: v32_params(args.bars, True) for ticker in tickers}
        if stall is not None:
            breakeven, bars, lock = stall
            params = {
                ticker: params[ticker].overrides(
                    stall_breakeven_atr=breakeven, stall_bars=bars, stall_lock_atr=lock
                )
                for ticker in tickers
            }
        signals = build_all_signals(metas, params, candles)
        total = sum(len(series) for series in signals.values())
        scope = "база" if stall is None else f"застой {stall[0]}x{stall[1]}"
        print(f"\nсигнальные потоки ({scope}): сделок {total}", flush=True)
        for case in group:
            settings = WalkForwardSettings(
                capital_rub=args.capital,
                risk_pct_per_trade=case["risk"],
                step_days=14,
                lookback_days=45,
                top=12,
                min_pf=1.15,
                min_trades=3,
                max_concurrent_positions=4,
                conviction_sizing=True,
                free_slot_on_breakeven=True,
                max_total_open_positions=8,
                conviction_leader_boost=1.45,
                throttle_start_pct=6.0,
                warmup_days=45,
                max_position_margin_pct=float(case["cap"]),
            )
            if "exclude" in case:
                settings.exclude_tickers = tuple(case["exclude"])
            result = run_walkforward(signals, metas, start_ms, end_ms, settings)
            metrics = result.metrics(args.capital)
            oos = sum(t.net_rub for t in result.trades if t.exit_ms < split_ms)
            is_net = sum(t.net_rub for t in result.trades if t.exit_ms >= split_ms)
            report["cases"][case["name"]] = {
                "params": {k: v for k, v in case.items() if k != "stall"},
                "stall": stall,
                "metrics": metrics,
                "oosNet": round(oos, 2),
                "isNet": round(is_net, 2),
                "rejected": {
                    "slots": result.rejected_by_slots,
                    "risk": result.rejected_by_risk,
                    "margin": result.rejected_by_margin,
                },
                "baskets": result.baskets,
            }
            print(
                f"{case['name']:<30} {metrics['trades']:>5} {metrics['trades'] / weeks:>8.1f} "
                f"{oos:>9.0f} {is_net:>9.0f} {metrics['netRub']:>9.0f} "
                f"{str(metrics['profitFactor']):>6} {str(metrics['sharpe']):>6} "
                f"{metrics['maxDrawdownPct']:>6} {metrics['annualizedPct']:>7.1f} "
                f"{result.rejected_by_margin:>7}",
                flush=True,
            )
            months = " ".join(f"{k[5:]}:{v:.0f}" for k, v in metrics["monthly"].items())
            print(f"    месяцы {months}", flush=True)

    out = Path(args.out or f"reports/forts_v44_walkforward_{args.bars}m.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено {out} за {time.time() - started:.0f} c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
