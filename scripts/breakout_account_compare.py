"""Сравнение схем капитала v3.2: изолированные мини-счета против общего счёта.

Скрипт отвечает на вопрос §2 задания v3.2: почему при делении 50 000 ₽ на три
изолированные части `USDRUBF` почти не торгует, и что даёт общий счёт с общей
маржой (`SwingAccountBacktester`).

Считает OOS / IS / Full по корзинам `CNYRUBF + GLDRUBF` и
`CNYRUBF + GLDRUBF + USDRUBF` для нескольких комбинаций (риск, утилизация ГО)
и складывает результат в `reports/forts_v32_final.json`.

Пример::

    py -3 scripts/breakout_account_compare.py
    py -3 scripts/breakout_account_compare.py --bars 30 --er 0.25
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

from bot_a.backtest.breakout_backtest import (  # noqa: E402
    BreakoutBacktester,
    BreakoutSettings,
    aggregate,
    portfolio_curve,
)
from bot_a.backtest.swing_portfolio import (  # noqa: E402
    SwingAccountBacktester,
    SwingAccountSettings,
)
from bot_a.economics.fees import Tariff  # noqa: E402
from bot_a.engine.breakout_fsm import BreakoutParams  # noqa: E402
from breakout_scan import load_candles, load_meta  # noqa: E402
from breakout_swing_scan import PERIODS, build_params  # noqa: E402

PERIODS_LIST = ("oos", "is", "full")
BASKETS: Dict[str, List[str]] = {
    "CNY+GLD": ["CNYRUBF", "GLDRUBF"],
    "CNY+GLD+USD": ["CNYRUBF", "GLDRUBF", "USDRUBF"],
}
# (корзина, схема, риск %, утилизация ГО %)
CASES = (
    ("CNY+GLD", "isolated", 1.5, 60.0),
    ("CNY+GLD", "shared", 1.0, 60.0),
    ("CNY+GLD+USD", "isolated", 1.5, 60.0),
    ("CNY+GLD+USD", "shared", 1.0, 60.0),
    ("CNY+GLD+USD", "shared", 0.75, 80.0),
    ("CNY+GLD+USD", "shared", 1.5, 80.0),
)


def _params(bars: int, er: float) -> BreakoutParams:
    params = build_params(bars, "bar_close", True, "run")
    return params.overrides(min_efficiency_ratio=er) if er > 0 else params


def _curve(per_ticker: Sequence[Any], capital: float) -> Dict[str, Any]:
    curve = portfolio_curve(list(per_ticker), capital)
    months = {key: round(value, 2) for key, value in curve["monthly"].items()}
    return {
        "sharpe": round(curve["sharpe"], 3) if curve.get("sharpe") is not None else None,
        "dd": round(curve["maxDrawdownPct"], 2),
        "monthly": months,
    }


def isolated(
    tickers: Sequence[str],
    params: BreakoutParams,
    capital: float,
    risk: float,
    candles: Dict[str, Dict[str, Sequence[Any]]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for period in PERIODS_LIST:
        share = capital / max(len(tickers), 1)
        settings = BreakoutSettings(
            capital_rub=share, risk_pct_per_trade=risk, tariff=Tariff.PREMIUM
        )
        results = [
            BreakoutBacktester(load_meta(ticker), params, settings).run(candles[period][ticker])
            for ticker in tickers
        ]
        agg = aggregate(results, capital)
        out[period] = {
            "net": round(agg["netRub"], 2),
            "pf": agg.get("profitFactor"),
            "trades": agg["trades"],
            **_curve(results, capital),
        }
    return out


def shared(
    tickers: Sequence[str],
    params: BreakoutParams,
    capital: float,
    risk: float,
    utilization: float,
    candles: Dict[str, Dict[str, Sequence[Any]]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for period in PERIODS_LIST:
        engine = SwingAccountBacktester(
            [(load_meta(ticker), params) for ticker in tickers],
            SwingAccountSettings(
                capital_rub=capital,
                risk_pct_per_trade=risk,
                max_margin_utilization_pct=utilization,
            ),
        )
        result = engine.run({ticker: candles[period][ticker] for ticker in tickers})
        profits = sum(trade.net_rub for trade in result.trades if trade.net_rub > 0)
        losses = -sum(trade.net_rub for trade in result.trades if trade.net_rub <= 0)
        out[period] = {
            "net": round(result.net_rub, 2),
            "pf": round(profits / losses, 3) if losses > 0 else None,
            "trades": len(result.trades),
            "maxMargin": round(result.max_margin_used, 0),
            "rejectedMargin": result.rejected_by_margin,
            "rejectedRisk": result.rejected_by_risk,
            "perTicker": {t: round(r.net_rub, 2) for t, r in result.per_ticker.items()},
            "perTickerTrades": {t: len(r.trades) for t, r in result.per_ticker.items()},
            **_curve(result.per_ticker.values(), capital),
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Изолированные счета против общего (v3.2)")
    parser.add_argument("--bars", type=int, default=30)
    parser.add_argument("--er", type=float, default=0.25)
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--out", default="reports/forts_v32_final.json")
    args = parser.parse_args(argv)

    params = _params(args.bars, args.er)
    tickers = sorted({ticker for basket in BASKETS.values() for ticker in basket})
    candles = {
        period: {ticker: load_candles(ticker, *PERIODS[period], args.bars) for ticker in tickers}
        for period in PERIODS_LIST
    }

    report: Dict[str, Any] = {}
    print(f"{'случай':<28} {'OOS':>9} {'IS':>9} {'Full':>9} {'PF':>6} {'Sh':>6} {'DD%':>6} {'сдел':>5}")
    for basket, scheme, risk, utilization in CASES:
        if scheme == "isolated":
            data = isolated(BASKETS[basket], params, args.capital, risk, candles)
            label = f"{basket}/isolated r={risk:g}"
        else:
            data = shared(BASKETS[basket], params, args.capital, risk, utilization, candles)
            label = f"{basket}/shared r={risk:g} u={utilization:g}"
        report[label] = data
        full = data["full"]
        print(
            f"{label:<28} {data['oos']['net']:>9.0f} {data['is']['net']:>9.0f} "
            f"{full['net']:>9.0f} {full['pf']:>6} {full['sharpe']:>6} {full['dd']:>6} {full['trades']:>5}"
        )
        if "perTicker" in full:
            print(
                f"{'':<28} по тикерам {full['perTicker']} сделок {full['perTickerTrades']} "
                f"пик ГО {full['maxMargin']:.0f} ₽, отказы: риск {full['rejectedRisk']}, "
                f"ГО {full['rejectedMargin']}"
            )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
