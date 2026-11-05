"""Сводка по JSON сетки v3.0: лучшая комбинация на инструмент по каждому периоду.

Пример::

    py -3 scripts/breakout_summary.py reports/forts_m15_scan.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else ROOT / "reports" / "forts_breakout_scan.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    print("сетка:", json.dumps(data["grid"], ensure_ascii=False))
    print("бары:", data.get("barMinutes", 1), "мин | капитал:", data.get("capital"), "RUB")
    for period in ("oos", "is", "full"):
        print(f"=== {period}")
        for ticker, blocks in sorted(data["results"].items()):
            block = blocks.get(period)
            if not block:
                continue
            rows = block["rows"]
            best = max(rows, key=lambda row: row["netRub"])
            with_trades = sum(1 for row in rows if row["trades"])
            print(
                f"{ticker:<9} свечей {block['candles']:>6} | лучшая {best['netRub']:>10.2f} RUB"
                f" сделок {best['trades']:>3} PF {str(best['profitFactor']):>7}"
                f" DD {best['maxDrawdownRub']:>9.2f} | комбинаций со сделками {with_trades}/{len(rows)}"
            )
            print(f"           параметры {best['params']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
