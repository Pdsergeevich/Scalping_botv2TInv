"""Портфель v4.0: инструменты и движки, отобранные ТОЛЬКО на OOS-периоде.

Протокол (против подгонки):

1. На OOS (`25.04–25.07`) для каждого инструмента считается три варианта движков
   (`A` — только пробой, `A+B` — плюс откаты, `A+C` — плюс фейды; см.
   `scripts/multiregime_scan.py`), и для инструмента выбирается лучший вариант
   при условии минимум 3 сделок.
2. Выбранный набор «замораживается» и прогоняется на IS (`25.07–05.11`) без
   каких-либо изменений — это слепой тест.
3. Риск-тиры и утилизация ГО перебираются на OOS, победитель проверяется на IS.

Такой протокол не даёт «подогнать» инструменты под будущее: IS не участвует в выборе.

Пример::

    py -3 scripts/multiregime_portfolio.py --bars 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bot_a.backtest.breakout_backtest import portfolio_curve  # noqa: E402
from bot_a.backtest.swing_portfolio import (  # noqa: E402
    SwingAccountBacktester,
    SwingAccountSettings,
)
from bot_a.engine.breakout_fsm import BreakoutParams  # noqa: E402
from breakout_scan import load_candles, load_meta  # noqa: E402
from breakout_swing_scan import PERIODS, build_params  # noqa: E402
from multiregime_scan import params_for, v32_params, with_engines  # noqa: E402

# Отбор берётся из `reports/forts_v40_selection_<bars>m.json`, который строит
# `scripts/multiregime_select.py`: инструмент → набор движков. Выбор делается
# только на OOS (обе половины в плюсе), IS — слепой тест.
SELECTION_FILE = "reports/forts_v40_selection_30m.json"
MIN_OOS_TRADES = 3


def params_for_selection(ticker: str, base: BreakoutParams, engine: str) -> BreakoutParams:
    """Движки строго по букве отбора: A — пробой, B — +откаты, C — +фейды, ABC — всё."""
    if engine == "BASE":
        return base
    if engine == "B":
        return with_engines(base, pullback=True)
    if engine == "C":
        return with_engines(base, fade=True)
    if engine == "ABC":
        return with_engines(base, pullback=True, fade=True)
    return base


def evaluate(
    tickers: Sequence[str],
    engines: Dict[str, str],
    base: BreakoutParams,
    capital: float,
    risk: float,
    utilization: float,
    candles: Dict[str, Dict[str, Sequence[Any]]],
    max_concurrent: int = 0,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for period in ("oos", "is", "full"):
        instruments = [
            (load_meta(ticker), params_for_selection(ticker, base, engines[ticker]))
            for ticker in tickers
        ]
        engine = SwingAccountBacktester(
            instruments,
            SwingAccountSettings(
                capital_rub=capital,
                risk_pct_per_trade=risk,
                max_margin_utilization_pct=utilization,
                max_concurrent_positions=max_concurrent,
            ),
        )
        result = engine.run({ticker: candles[period][ticker] for ticker in tickers})
        profits = sum(trade.net_rub for trade in result.trades if trade.net_rub > 0)
        losses = -sum(trade.net_rub for trade in result.trades if trade.net_rub <= 0)
        curve = portfolio_curve(list(result.per_ticker.values()), capital)
        out[period] = {
            "net": round(result.net_rub, 2),
            "pf": round(profits / losses, 3) if losses > 0 else None,
            "trades": len(result.trades),
            "sharpe": round(curve["sharpe"], 3) if curve.get("sharpe") is not None else None,
            "dd": round(curve["maxDrawdownPct"], 2),
            "monthly": {key: round(value, 2) for key, value in curve["monthly"].items()},
            "fees": round(result.fees_rub, 2),
            "maxMargin": round(result.max_margin_used, 0),
            "rejectedMargin": result.rejected_by_margin,
            "rejectedRisk": result.rejected_by_risk,
            "rejectedSlots": result.rejected_by_slots,
            "perTicker": {t: round(r.net_rub, 2) for t, r in result.per_ticker.items()},
            "perTickerTrades": {t: len(r.trades) for t, r in result.per_ticker.items()},
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Портфель v4.0 (отбор на OOS, тест на IS)")
    parser.add_argument("--bars", type=int, default=30)
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--out", default="")
    parser.add_argument("--selection", default="")
    args = parser.parse_args(argv)

    selection_path = Path(args.selection or SELECTION_FILE.replace("30m", f"{args.bars}m"))
    if not selection_path.exists():
        print(f"нет файла отбора {selection_path}: сначала запустите multiregime_select.py")
        return 2
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    engines = {
        ticker: engine
        for ticker, engine in payload["selection"].items()
        if any((ROOT / "data_cache").glob(f"iss_forts_{ticker}_*.json"))
    }
    tickers = list(engines)
    print(f"портфель v4.0 ({len(tickers)}): " + ", ".join(f"{t}:{engines[t]}" for t in tickers), flush=True)

    started = time.time()
    candles: Dict[str, Dict[str, Sequence[Any]]] = {}
    for period in ("oos", "is", "full"):
        stage = {ticker: load_candles(ticker, *PERIODS[period], args.bars) for ticker in tickers}
        candles[period] = stage
        print(f"  данные {period}: {sum(len(v) for v in stage.values())} баров", flush=True)

    base = v32_params(args.bars, True)
    report: Dict[str, Any] = {"bars": args.bars, "capital": args.capital, "selection": engines, "cases": {}}
    print(
        f"\n{'случай':<24} {'сдел':>5} {'сдел/нед':>8} {'OOS':>9} {'IS':>9} {'Full':>9} "
        f"{'PF':>6} {'Sh':>6} {'DD%':>6} {'год%':>7} {'ГО':>8}",
        flush=True,
    )
    core = [ticker for ticker in tickers if ticker not in {"GOLDM_CONT", "Eu_CONT", "Si_CONT", "VTBR_CONT"}]
    cases: List[Dict[str, Any]] = [
        {"name": "v3.2 эталон (2 тикера)", "tickers": ["CNYRUBF", "GLDRUBF"],
         "engines": {"CNYRUBF": "BASE", "GLDRUBF": "BASE"}, "risk": 1.0, "util": 60.0,
         "concurrent": 0},
    ]
    for risk, util, slots in ((0.5, 75.0, 3), (0.8, 75.0, 3), (1.0, 75.0, 0), (1.0, 75.0, 3), (1.5, 85.0, 3)):
        cases.append(
            {"name": f"v4.0 все r={risk:.1f}% ГО={util:.0f}% слотов={slots or 'inf'}",
             "tickers": tickers, "engines": engines, "risk": risk, "util": util,
             "concurrent": slots}
        )
    for risk, slots in ((0.5, 3), (0.8, 3), (1.0, 3), (1.5, 3), (2.0, 4)):
        cases.append(
            {"name": f"v4.0 ядро r={risk:.1f}% слотов={slots}", "tickers": core,
             "engines": {ticker: engines[ticker] for ticker in core}, "risk": risk,
             "util": 85.0, "concurrent": slots}
        )

    for case in cases:
        data = evaluate(
            case["tickers"], case["engines"], base, args.capital, case["risk"], case["util"],
            candles, case.get("concurrent", 0),
        )
        report["cases"][case["name"]] = {
            "risk": case["risk"], "utilization": case["util"], "tickers": list(case["tickers"]),
            "concurrent": case.get("concurrent", 0),
            "engines": case["engines"], "periods": data,
        }
        full = data["full"]
        annual = full["net"] / args.capital * 100.0 * 365.0 / 194.0
        print(
            f"{case['name']:<34} {full['trades']:>5} {full['trades'] / (194 / 7):>8.1f} "
            f"{data['oos']['net']:>9.0f} {data['is']['net']:>9.0f} {full['net']:>9.0f} "
            f"{str(full['pf']):>6} {str(full['sharpe']):>6} {full['dd']:>6} {annual:>7.1f} "
            f"{full['maxMargin']:>8.0f}",
            flush=True,
        )
        print("    месяцы " + " ".join(f"{k[5:]}:{v:.0f}" for k, v in full["monthly"].items()), flush=True)
        print(
            "    по тикерам " + " ".join(
                f"{t}:{v:.0f}({full['perTickerTrades'][t]})" for t, v in
                sorted(full["perTicker"].items(), key=lambda kv: -kv[1])
            ),
            flush=True,
        )

    out = Path(args.out or f"reports/forts_v40_portfolio_{args.bars}m.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено {out} за {time.time() - started:.0f} c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
