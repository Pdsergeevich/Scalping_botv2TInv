"""Портфельный прогон v3.0 на фиксированном наборе параметров (без подгонки по инструментам).

Пример::

    py -3 scripts/breakout_portfolio.py --tickers CNYRUBF CRZ6 MMZ6 --bar-minutes 15 \
        --params '{"donchian_bars":20,"initial_stop_atr_multiple":1.5}'

Капитал делится поровну между инструментами; итог складывается по времени выхода сделок,
поэтому просадка считается по объединённой кривой.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

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
from breakout_scan import PERIODS, load_candles, load_meta  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Портфель v3.0 (FORTS breakout)")
    parser.add_argument("--tickers", nargs="*", default=["CNYRUBF", "CRZ6", "MMZ6", "GLDRUBF"])
    parser.add_argument("--bar-minutes", type=int, default=15)
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--risk", type=float, default=1.5)
    parser.add_argument("--periods", nargs="*", default=["full", "oos", "is"])
    parser.add_argument(
        "--params",
        default="{}",
        help="JSON с переопределением BreakoutParams (snake_case или camelCase)",
    )
    parser.add_argument("--params-file", dest="params_file", default=None, help="JSON-файл с параметрами")
    parser.add_argument("--trades", action="store_true", help="печатать сделки")
    args = parser.parse_args(argv)

    overrides = json.loads(args.params) if args.params else {}
    if args.params_file:
        overrides.update(json.loads(Path(args.params_file).read_text(encoding="utf-8")))
    params = BreakoutParams.from_mapping(overrides)
    share = args.capital / max(len(args.tickers), 1)
    payload = {"params": params.to_dict(), "barMinutes": args.bar_minutes, "periods": {}}
    print(
        f"Параметры: N={params.donchian_bars} стоп={params.initial_stop_atr_multiple} "
        f"трейл={params.trail_atr_multiple} объём={params.volume_multiple} "
        f"сжатие={params.max_squeeze_ratio} расширение={params.max_entry_extension_atr} "
        f"BE={params.breakeven_trigger_atr}/{params.breakeven_lock_atr} "
        f"| бары {args.bar_minutes} мин | капитал {args.capital:.0f} RUB"
    )
    for period in args.periods:
        results = []
        print(f"\n=== {period}")
        print(
            f"{'Инструмент':<10} {'сделок':>7} {'net RUB':>10} {'PF':>7} {'WinRate':>8} "
            f"{'DD RUB':>9} {'ann.%':>8}"
        )
        for ticker in args.tickers:
            meta = load_meta(ticker)
            candles = load_candles(ticker, *PERIODS[period], args.bar_minutes)
            if not candles:
                continue
            backtester = BreakoutBacktester(
                meta,
                params,
                BreakoutSettings(capital_rub=share, risk_pct_per_trade=args.risk, tariff=Tariff.PREMIUM),
            )
            result = backtester.run(candles)
            results.append(result)
            summary = result.summary(share)
            pf = summary["profitFactor"]
            print(
                f"{ticker:<10} {summary['trades']:>7} {summary['netRub']:>10.2f} "
                f"{('∞' if pf == float('inf') else (f'{pf:.2f}' if pf else '—')):>7} "
                f"{float(summary['winRate']) * 100:>7.1f}% {summary['maxDrawdownRub']:>9.2f} "
                f"{summary['annualizedPct']:>8.2f}"
            )
            if args.trades:
                for trade in result.trades:
                    print("   ", trade.to_dict())
        combined = aggregate(results, args.capital)
        pf = combined["profitFactor"]
        print(
            f"{'ИТОГО':<10} {combined['trades']:>7} {combined['netRub']:>10.2f} "
            f"{('∞' if pf == float('inf') else (f'{pf:.2f}' if pf else '—')):>7} "
            f"{float(combined['winRate']) * 100:>7.1f}% {combined['maxDrawdownRub']:>9.2f} "
            f"{combined['annualizedPct']:>8.2f}"
        )
        payload["periods"][period] = {
            "perTicker": [result.summary(share) for result in results],
            "portfolio": combined,
        }
    out = ROOT / "reports" / "forts_portfolio.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
