"""Портфель v3.1: единый набор параметров на FX/товарную корзину + кривая эквити 50 000 ₽.

Капитал делится поровну между инструментами (каждый считает сайзинг от своей доли),
затем сделки объединяются по времени выхода: Net, PF, Sharpe по дням, Max DD в рублях и %.

Пример::

    py -3 scripts/breakout_swing_portfolio.py --bar-minutes 30 --variant run --overnight
    py -3 scripts/breakout_swing_portfolio.py --tickers CNYRUBF GLDRUBF --timeframe 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bot_a.backtest.breakout_backtest import (  # noqa: E402
    BreakoutBacktester,
    BreakoutResult,
    BreakoutSettings,
    aggregate,
    portfolio_curve,
)
from bot_a.economics.fees import Tariff  # noqa: E402
from breakout_scan import load_candles, load_meta  # noqa: E402
from breakout_swing_scan import DEFAULT_TICKERS, PERIODS, build_params  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Портфель v3.1 (FX + товары, свинг)")
    parser.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    parser.add_argument("--bar-minutes", type=int, default=30)
    parser.add_argument("--entry-mode", default="bar_close", choices=["bar_close", "stop_level"])
    parser.add_argument("--variant", default="run")
    parser.add_argument("--overnight", action="store_true", help="перенос позиции через ночь")
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--risk", type=float, default=1.5)
    parser.add_argument("--trades", action="store_true", help="печатать сделки")
    parser.add_argument("--out", default="reports/forts_swing_portfolio.json")
    args = parser.parse_args(argv)

    params = build_params(args.bar_minutes, args.entry_mode, args.overnight, args.variant)
    share = args.capital / max(len(args.tickers), 1)
    settings = BreakoutSettings(
        capital_rub=share, risk_pct_per_trade=args.risk, tariff=Tariff.PREMIUM
    )
    print(
        f"Параметры: бары {args.bar_minutes} мин, вход {args.entry_mode}, "
        f"ночь {'да' if args.overnight else 'нет'}, вариант {args.variant}\n"
        f"N={params.donchian_bars} стоп={params.initial_stop_atr_multiple} "
        f"трейл={params.trail_atr_multiple}/{params.trail_trigger_atr} "
        f"BE={params.breakeven_trigger_atr}/{params.breakeven_lock_atr} "
        f"тейк={params.take_profit_atr} объём×{params.volume_multiple} "
        f"сжатие={params.max_squeeze_ratio} | капитал {args.capital:.0f} ₽ "
        f"({share:.0f} ₽ на инструмент)"
    )
    payload: Dict[str, object] = {
        "params": params.to_dict(),
        "barMinutes": args.bar_minutes,
        "entryMode": args.entry_mode,
        "overnight": args.overnight,
        "variant": args.variant,
        "capital": args.capital,
        "riskPct": args.risk,
        "tickers": args.tickers,
        "periods": {},
    }
    all_ok = True
    for period in ("oos", "is", "full"):
        results: List[BreakoutResult] = []
        print(f"\n=== {period}")
        print(
            f"{'Инструмент':<10} {'сделок':>7} {'net ₽':>10} {'PF':>7} {'WR':>7} "
            f"{'DD ₽':>9} {'удерж,ч':>8}"
        )
        for ticker in args.tickers:
            candles = load_candles(ticker, *PERIODS[period], args.bar_minutes)
            if not candles:
                continue
            meta = load_meta(ticker)
            result = BreakoutBacktester(meta, params, settings).run(candles)
            results.append(result)
            summary = result.summary(share)
            pf = summary["profitFactor"]
            print(
                f"{ticker:<10} {summary['trades']:>7} {summary['netRub']:>10.2f} "
                f"{('∞' if pf == float('inf') else (f'{pf:.2f}' if pf else '—')):>7} "
                f"{float(summary['winRate']) * 100:>6.1f}% {summary['maxDrawdownRub']:>9.2f} "
                f"{float(summary['avgHoldMinutes']) / 60.0:>8.1f}"
            )
            if args.trades and period == "full":
                for trade in result.trades:
                    print("   ", trade.to_dict())
        combined = aggregate(results, args.capital)
        curve = portfolio_curve(results, args.capital)
        pf = combined["profitFactor"]
        years = max((combined["bars"] and 0) or 0, 0)
        total_days = curve["days"]
        print(
            f"{'ИТОГО':<10} {combined['trades']:>7} {combined['netRub']:>10.2f} "
            f"{('∞' if pf == float('inf') else (f'{pf:.2f}' if pf else '—')):>7} "
            f"{float(combined['winRate']) * 100:>6.1f}% {curve['maxDrawdownRub']:>9.2f} "
            f"{float(combined['avgHoldMinutes']) / 60.0:>8.1f}"
        )
        print(
            f"  кривая: Sharpe {curve['sharpe']} | Max DD {curve['maxDrawdownRub']:.2f} ₽ "
            f"({curve['maxDrawdownPct']:.2f}%) | дней с PnL {total_days} | "
            f"лучший день {curve['bestDay']:+.2f} | худший {curve['worstDay']:+.2f} | "
            f"годовая {combined['annualizedPct']:+.2f}%"
        )
        if curve.get("monthly"):
            print(
                "  месяцы: "
                + " ".join(f"{month}:{value:+.0f}" for month, value in curve["monthly"].items())
            )
        if period in ("oos", "is") and float(combined["netRub"]) <= 0:
            all_ok = False
        payload["periods"][period] = {
            "perTicker": [result.summary(share) for result in results],
            "portfolio": combined,
            "curve": curve,
        }
    print(
        f"\nЕдиный набор параметров прибылен и в OOS, и в IS: "
        f"{'ДА' if all_ok else 'НЕТ'}"
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"сохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
