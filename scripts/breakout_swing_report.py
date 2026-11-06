"""Сводка свипа v3.1: сравнение таймфреймов, режимов входа и переноса через ночь.

Ищет **единый** набор параметров, положительный одновременно в OOS и IS (без подгонки
по инструментам), и печатает таблицы для отчёта.

Пример::

    py -3 scripts/breakout_swing_report.py
    py -3 scripts/breakout_swing_report.py --basket CNYRUBF GLDRUBF
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PERIODS = ("oos", "is", "full")


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and value == float("inf"):
        return "∞"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def series_index(scan: Dict[str, Any]) -> Dict[tuple, Dict[str, Dict[str, Any]]]:
    """(barMinutes, entryMode, overnight, variant) → {ticker: {period: summary}}."""
    index: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
    for ticker, rows in scan["results"].items():
        for row in rows:
            key = (row["barMinutes"], row["entryMode"], row["overnight"], row["variant"])
            index.setdefault(key, {})[ticker] = {
                period: row.get(period) for period in PERIODS
            }
    return index


def basket_totals(
    index_entry: Dict[str, Dict[str, Any]], basket: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, float]] = {}
    for period in PERIODS:
        net = 0.0
        trades = 0
        wins = 0
        for ticker in basket:
            summary = index_entry.get(ticker, {}).get(period)
            if not summary:
                continue
            net += float(summary["netRub"])
            trades += int(summary["trades"])
            wins += int(round(float(summary["winRate"]) * int(summary["trades"])))
        totals[period] = {
            "netRub": net,
            "trades": trades,
            "winRate": (wins / trades) if trades else 0.0,
        }
    return totals


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Отчёт по свипу v3.1")
    parser.add_argument("--scan", default="reports/forts_swing_scan.json")
    parser.add_argument("--basket", nargs="*", default=None)
    parser.add_argument("--section", default="all")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    scan = load(ROOT / args.scan)
    index = series_index(scan)
    basket = args.basket or sorted(scan["results"].keys())
    blocks: List[str] = []

    if args.section in ("all", "grid"):
        for period in PERIODS:
            blocks.append(f"### Период {period}: корзина {', '.join(basket)}\n")
            blocks.append(
                "| Бары | Вход | Ночь | Выход | Сделок | Net, ₽ | WinRate | WinRate по инструментам |"
            )
            blocks.append("|---|---|---|---|---|---|---|---|")
            rows = []
            for key, per_ticker in index.items():
                totals = basket_totals(per_ticker, basket)
                item = totals[period]
                rows.append((item["netRub"], key, item))
            rows.sort(key=lambda item: -item[0])
            for net, key, item in rows:
                bar_minutes, entry_mode, overnight, variant = key
                per_instrument = " ".join(
                    f"{ticker}:{_fmt((per_ticker.get(period) or {}).get('netRub'))}"
                    for ticker, per_ticker in sorted(index[key].items())
                )
                blocks.append(
                    f"| {bar_minutes} | {entry_mode} | {'да' if overnight else 'нет'} | {variant} | "
                    f"{item['trades']} | {net:+.2f} | {item['winRate'] * 100:.1f}% | {per_instrument} |"
                )
            blocks.append("")

    if args.section in ("all", "dual"):
        blocks.append("### Комбинации, положительные и в OOS, и в IS\n")
        blocks.append("| Бары | Вход | Ночь | Выход | OOS, ₽ | IS, ₽ | Full, ₽ | Сделок Full |")
        blocks.append("|---|---|---|---|---|---|---|---|")
        found = 0
        for key, per_ticker in index.items():
            totals = basket_totals(per_ticker, basket)
            if totals["oos"]["netRub"] > 0 and totals["is"]["netRub"] > 0:
                bar_minutes, entry_mode, overnight, variant = key
                found += 1
                blocks.append(
                    f"| {bar_minutes} | {entry_mode} | {'да' if overnight else 'нет'} | {variant} | "
                    f"{totals['oos']['netRub']:+.2f} | {totals['is']['netRub']:+.2f} | "
                    f"{totals['full']['netRub']:+.2f} | {totals['full']['trades']} |"
                )
        if not found:
            blocks.append("| — | — | — | — | — | — | — | — |")
        blocks.append("")

    if args.section in ("all", "perpair"):
        blocks.append("### Лучшая комбинация на инструмент (по Full, для справки)\n")
        blocks.append("| Инструмент | Бары | Вход | Ночь | Выход | OOS, ₽ | IS, ₽ | Full, ₽ | PF Full | DD Full |")
        blocks.append("|---|---|---|---|---|---|---|---|---|---|")
        for ticker in basket:
            best = None
            for key, per_ticker in index.items():
                summary = per_ticker.get(ticker, {}).get("full")
                if not summary:
                    continue
                if best is None or float(summary["netRub"]) > float(best[1]["netRub"]):
                    best = (key, summary, per_ticker.get(ticker, {}))
            if best is None:
                continue
            key, full, per = best
            bar_minutes, entry_mode, overnight, variant = key
            blocks.append(
                f"| {ticker} | {bar_minutes} | {entry_mode} | {'да' if overnight else 'нет'} | {variant} | "
                f"{_fmt((per.get('oos') or {}).get('netRub'))} | {_fmt((per.get('is') or {}).get('netRub'))} | "
                f"{full['netRub']:+.2f} | {_fmt(full['profitFactor'])} | {full['maxDrawdownRub']:.2f} |"
            )
        blocks.append("")

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
