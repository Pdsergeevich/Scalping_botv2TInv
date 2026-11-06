"""Отбор «инструмент × движок» ТОЛЬКО на OOS-периоде (v4.0).

Для каждого инструмента и каждого варианта движков считается изолированный бэктест
(`A` — только пробой, `B` — пробой + откаты, `C` — пробой + фейды, `ABC` — все три),
берётся результат на OOS (`25.04–25.07`) и проверяется на IS (`25.07–05.11`).

Правило отбора (задано заранее, без просмотра IS):

* OOS Net > 0 и не меньше `--min-trades` сделок;
* из прошедших берётся вариант с максимальным OOS Net.

Результат — `reports/forts_v40_selection.json`, который использует
`scripts/multiregime_portfolio.py`. IS в отборе не участвует: это слепой тест.

Пример::

    py -3 scripts/multiregime_select.py --bars 30
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

from bot_a.backtest.breakout_backtest import BreakoutBacktester, BreakoutSettings  # noqa: E402
from bot_a.economics.fees import Tariff  # noqa: E402
from bot_a.engine.breakout_fsm import BreakoutParams  # noqa: E402
from breakout_scan import load_candles, load_meta  # noqa: E402
from breakout_swing_scan import PERIODS  # noqa: E402
from multiregime_scan import PERIODS as _SCAN_PERIODS  # noqa: E402,F401
from multiregime_scan import available_continuous, v32_params, with_engines  # noqa: E402

PERPETUALS = ("CNYRUBF", "GLDRUBF", "USDRUBF", "EURRUBF", "IMOEXF", "GAZPF")
ENGINES = ("A", "B", "C", "ABC")
# Две половины OOS: и обе должны быть в плюсе — иначе прибыль держится на одном
# всплеске (это и есть главный признак подгонки, обнаруженный в отчёте §3).
HALVES = {
    "oos1": ("2026-04-25", "2026-06-05"),
    "oos2": ("2026-06-05", "2026-07-25"),
}


def engine_params(ticker: str, base: BreakoutParams, engine: str) -> BreakoutParams:
    if engine == "A":
        return base
    if engine == "B":
        return with_engines(base, pullback=True)
    if engine == "C":
        return with_engines(base, fade=True)
    return with_engines(base, pullback=True, fade=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="OOS-отбор инструментов и движков (v4.0)")
    parser.add_argument("--bars", type=int, default=30)
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--risk", type=float, default=1.0)
    parser.add_argument("--min-trades", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    cont = available_continuous()
    tickers = [t for t in PERPETUALS if any((ROOT / "data_cache").glob(f"iss_forts_{t}_*.json"))]
    tickers += sorted(cont.values())
    print(f"инструментов: {len(tickers)}", flush=True)

    base = v32_params(args.bars, True)
    data: Dict[str, Dict[str, Sequence[Any]]] = {}
    for period in ("oos", "is", "full"):
        data[period] = {t: load_candles(t, *PERIODS[period], args.bars) for t in tickers}
    for half, bounds in HALVES.items():
        data[half] = {t: load_candles(t, *bounds, args.bars) for t in tickers}

    table: Dict[str, Any] = {}
    print(f"\n{'тикер':<12}" + "".join(f"{eng:>16}" for eng in ENGINES), flush=True)
    for ticker in tickers:
        meta = load_meta(ticker)
        row: Dict[str, Any] = {}
        line = f"{ticker:<12}"
        for engine in ENGINES:
            params = engine_params(ticker, base, engine)
            settings = BreakoutSettings(
                capital_rub=args.capital, risk_pct_per_trade=args.risk, tariff=Tariff.PREMIUM
            )
            oos = BreakoutBacktester(meta, params, settings).run(data["oos"][ticker])
            is_ = BreakoutBacktester(meta, params, settings).run(data["is"][ticker])
            full = BreakoutBacktester(meta, params, settings).run(data["full"][ticker])
            first = BreakoutBacktester(meta, params, settings).run(data["oos1"][ticker])
            second = BreakoutBacktester(meta, params, settings).run(data["oos2"][ticker])
            row[engine] = {
                "oos1": round(first.net_rub, 2),
                "oos2": round(second.net_rub, 2),
                "oos": round(oos.net_rub, 2),
                "oos_trades": len(oos.trades),
                "is": round(is_.net_rub, 2),
                "is_trades": len(is_.trades),
                "full": round(full.net_rub, 2),
                "full_trades": len(full.trades),
                "pf": round(full.profit_factor, 3) if full.profit_factor else None,
            }
            line += f"{row[engine]['oos']:>9.0f}({row[engine]['oos_trades']:>2})"
        table[ticker] = row
        print(line, flush=True)

    # (1) «Жадный» отбор по максимуму OOS — так делать нельзя, он переобучается;
    #     он остаётся в отчёте как контрольный пример.
    greedy: Dict[str, str] = {}
    for ticker, row in table.items():
        candidates = [
            (engine, stats)
            for engine, stats in row.items()
            if stats["oos"] > 0 and stats["oos_trades"] >= args.min_trades
        ]
        if candidates:
            greedy[ticker] = max(candidates, key=lambda item: item[1]["oos"])[0]

    # (2) Устойчивый отбор: прибыль в ОБЕИХ половинах OOS и минимум сделок.
    selection: Dict[str, str] = {}
    for ticker, row in table.items():
        candidates = [
            (engine, stats)
            for engine, stats in row.items()
            if stats["oos1"] > 0
            and stats["oos2"] > 0
            and stats["oos"] > 0
            and stats["oos_trades"] >= args.min_trades
        ]
        if candidates:
            selection[ticker] = max(candidates, key=lambda item: item[1]["oos"])[0]

    print("\nустойчивый отбор (обе половины OOS в плюсе → слепой IS):")
    oos_sum = is_sum = 0.0
    for ticker, engine in selection.items():
        stats = table[ticker][engine]
        oos_sum += stats["oos"]
        is_sum += stats["is"]
        print(
            f"  {ticker:<12} {engine:<4} OOS {stats['oos']:>8.0f} ({stats['oos_trades']:>3} сд.)"
            f"   IS {stats['is']:>8.0f} ({stats['is_trades']:>3} сд.)"
        )
    print(f"  ИТОГО по изолированным: OOS {oos_sum:.0f} ₽ / IS {is_sum:.0f} ₽")

    payload = {
        "bars": args.bars,
        "capital": args.capital,
        "risk": args.risk,
        "min_trades": args.min_trades,
        "table": table,
        "selection": selection,
        "greedy_oos": greedy,
    }
    out = Path(args.out or f"reports/forts_v40_selection_{args.bars}m.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    greedy_oos = sum(table[t][e]["oos"] for t, e in greedy.items())
    greedy_is = sum(table[t][e]["is"] for t, e in greedy.items())
    print(f"\nконтроль (жадный max OOS, {len(greedy)} инстр.): OOS {greedy_oos:.0f} ₽ / IS {greedy_is:.0f} ₽")
    print(f"сохранено {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
