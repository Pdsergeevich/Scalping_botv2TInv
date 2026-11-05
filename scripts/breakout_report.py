"""Аналитика и таблицы для отчёта v3.0 (OOS / IS / Full + портфель на 50 000 RUB).

Читает JSON сетки (`scripts/breakout_scan.py`) и пересчитывает нужные комбинации,
чтобы получить готовые к вставке в markdown таблицы:

* покрытие данных по инструментам;
* лучшие комбинации по OOS / IS / Full;
* портфель: капитал делится между инструментами, деньги складываются по времени выхода.

Пример::

    py -3 scripts/breakout_report.py --top 3
    py -3 scripts/breakout_report.py --portfolio CNYRUBF IMOEXF GLDRUBF --capital 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

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
from breakout_scan import PERIODS, build_params, load_candles, load_meta  # noqa: E402


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value == float("inf"):
            return "∞"
        return f"{value:.2f}"
    return str(value)


def coverage(tickers: Sequence[str]) -> List[str]:
    lines = [
        "| Инструмент | Свечей | Дней | Первый бар | Последний бар | ГО, RUB | Шаг | Шаг, RUB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ticker in tickers:
        meta = load_meta(ticker)
        candles = load_candles(ticker, *PERIODS["full"])
        if not candles:
            lines.append(f"| {ticker} | 0 | 0 | — | — | {meta.initial_margin_buy} | — | — |")
            continue
        first = datetime.fromtimestamp(candles[0].time_ms / 1000, tz=MSK)
        last = datetime.fromtimestamp(candles[-1].time_ms / 1000, tz=MSK)
        days = len({datetime.fromtimestamp(c.time_ms / 1000, tz=MSK).date() for c in candles})
        lines.append(
            f"| {ticker} | {len(candles)} | {days} | {first:%d.%m.%Y} | {last:%d.%m.%Y} | "
            f"{meta.initial_margin_buy} | {meta.min_price_increment} | "
            f"{meta.min_price_increment_amount} |"
        )
    return lines


def grid_table(scan: Dict[str, Any], period: str, top: int = 3) -> List[str]:
    lines = [
        "| Инструмент | # | N | Стоп, ATR | Трейл, ATR | Объём × | Сжатие | Сделок | Net, RUB | PF | WinRate | Max DD, RUB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ticker, periods in sorted(scan["results"].items()):
        block = periods.get(period)
        if not block:
            continue
        rows = sorted(block["rows"], key=lambda item: -float(item["netRub"]))[:top]
        for index, row in enumerate(rows, start=1):
            params = row["params"]
            squeeze = params["max_squeeze_ratio"]
            lines.append(
                f"| {ticker} | {index} | {int(params['donchian_bars'])} | "
                f"{params['initial_stop_atr_multiple']} | {params['trail_atr_multiple']} | "
                f"{params['volume_multiple']} | "
                f"{'∞' if float(squeeze) == 0.0 else squeeze} | {row['trades']} | "
                f"{row['netRub']:+.2f} | {_fmt(row['profitFactor'])} | "
                f"{float(row['winRate']) * 100:.0f}% | {row['maxDrawdownRub']:.2f} |"
            )
    return lines


def run_params(
    ticker: str,
    params: BreakoutParams,
    capital: float,
    risk_pct: float = 1.5,
    period: str = "full",
):
    meta = load_meta(ticker)
    candles = load_candles(ticker, *PERIODS[period])
    settings = BreakoutSettings(
        capital_rub=capital, risk_pct_per_trade=risk_pct, tariff=Tariff.PREMIUM
    )
    backtester = BreakoutBacktester(meta, params, settings)
    # Отдельные экземпляры считаются независимо: проскальзывание и комиссии — свои.
    return backtester.run(candles)


def best_params(scan: Dict[str, Any], ticker: str, period: str, min_trades: int = 3):
    rows = scan["results"][ticker][period]["rows"]
    usable = [row for row in rows if row["trades"] >= min_trades]
    if not usable:
        usable = rows
    best = max(usable, key=lambda row: float(row["netRub"]))
    return build_params(best["params"]), best


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Таблицы отчёта v3.0")
    parser.add_argument("--scan", default="reports/forts_breakout_scan.json")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--portfolio", nargs="*", default=None)
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--risk", type=float, default=1.5)
    parser.add_argument("--section", default="all")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    scan = json.loads((ROOT / args.scan).read_text(encoding="utf-8"))
    tickers = args.tickers or sorted(scan["results"].keys())
    blocks: List[str] = []

    if args.section in ("all", "coverage"):
        blocks.append("### Покрытие данных\n")
        blocks.extend(coverage(tickers))
        blocks.append("")

    for period in ("oos", "is", "full"):
        if args.section not in ("all", period):
            continue
        blocks.append(f"### Сетка, период {period}\n")
        blocks.extend(grid_table(scan, period, top=args.top))
        blocks.append("")

    if args.section in ("all", "portfolio"):
        portfolio_tickers = args.portfolio or tickers
        share = args.capital / max(len(portfolio_tickers), 1)
        results = []
        lines = [
            f"### Портфель {', '.join(portfolio_tickers)} (капитал {args.capital:.0f} RUB, "
            f"по {share:.0f} RUB на инструмент, период full)\n",
            "| Инструмент | Параметры (N/стоп/трейл/объём/сжатие) | Сделок | Net, RUB | PF | Max DD, RUB |",
            "|---|---|---|---|---|---|",
        ]
        for ticker in portfolio_tickers:
            params, _ = best_params(scan, ticker, "full")
            result = run_params(ticker, params, share, args.risk, "full")
            results.append(result)
            summary = result.summary(share)
            lines.append(
                f"| {ticker} | {params.donchian_bars}/{params.initial_stop_atr_multiple}/"
                f"{params.trail_atr_multiple}/{params.volume_multiple}/"
                f"{'∞' if params.max_squeeze_ratio == 0 else params.max_squeeze_ratio} | "
                f"{summary['trades']} | {summary['netRub']:+.2f} | {_fmt(summary['profitFactor'])} | "
                f"{summary['maxDrawdownRub']:.2f} |"
            )
        combined = aggregate(results, args.capital)
        lines.append(
            f"| **Итого** | — | {combined['trades']} | **{combined['netRub']:+.2f}** | "
            f"{_fmt(combined['profitFactor'])} | {combined['maxDrawdownRub']:.2f} |"
        )
        lines.append("")
        blocks.extend(lines)

    text = "\n".join(blocks)
    if args.out:
        path = ROOT / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"сохранено: {path}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
