"""Абляция v3.2: фильтр «пилы» (ER / наклон EMA) и двухступенчатый breakeven.

Скрипт гоняет набор вариантов параметров по корзине инструментов и печатает
OOS / IS / Full **и разбивку по месяцам** — главное, что нужно проверить:
сжимается ли летняя просадка (июнь–сентябрь), не теряя майскую и октябрьскую волны.

Режимы счёта:

* ``isolated`` — каждому инструменту своя доля капитала (как в v3.1);
* ``shared``   — один счёт с общей маржой (`bot_a/backtest/swing_portfolio.py`).

Пример::

    py -3 scripts/breakout_chop_scan.py --mode isolated
    py -3 scripts/breakout_chop_scan.py --mode shared --risk 1.0 --utilization 60
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
from bot_a.utils.time_provider import MSK  # noqa: E402
from breakout_scan import load_candles, load_meta  # noqa: E402
from breakout_swing_scan import BASE_PARAMS, EXIT_VARIANTS, PERIODS, build_params  # noqa: E402

DEFAULT_BASKET = ("CNYRUBF", "GLDRUBF", "USDRUBF")

# Варианты v3.2 поверх базового набора `run`.
VARIANTS: Dict[str, Dict[str, Any]] = {
    "v31": {},  # базовая линия v3.1
    "er25": {"min_efficiency_ratio": 0.25},
    "er15": {"min_efficiency_ratio": 0.15},
    "er18": {"min_efficiency_ratio": 0.18},
    "er20": {"min_efficiency_ratio": 0.20},
    "er22": {"min_efficiency_ratio": 0.22},
    "er28": {"min_efficiency_ratio": 0.28},
    "er30": {"min_efficiency_ratio": 0.30},
    "er35": {"min_efficiency_ratio": 0.35},
    "er40": {"min_efficiency_ratio": 0.40},
    "er45": {"min_efficiency_ratio": 0.45},
    "er20_slope15": {"min_efficiency_ratio": 0.20, "min_ema_slope_atr": 0.15},
    "er25_slope15": {"min_efficiency_ratio": 0.25, "min_ema_slope_atr": 0.15},
    "er25_so25": {
        "min_efficiency_ratio": 0.25,
        "scale_out_atr": 2.5,
        "scale_out_fraction": 0.5,
        "scale_out_stop_atr": 1.0,
    },
    "er25_bars30": {"min_efficiency_ratio": 0.25, "efficiency_ratio_bars": 30},
    # Интрадей-контроль для проверки эффекта переноса через ночь на исправленных данных.
    "intraday": {
        "allow_overnight_hold": False,
        "session_windows": "10:15-13:45,14:10-18:20",
        "trade_until": "18:35",
    },
    "intraday_er25": {
        "allow_overnight_hold": False,
        "session_windows": "10:15-13:45,14:10-18:20",
        "trade_until": "18:35",
        "min_efficiency_ratio": 0.25,
    },
    "slope15": {"min_ema_slope_atr": 0.15},
    "slope25": {"min_ema_slope_atr": 0.25},
    "soft16": {"soft_breakeven_trigger_atr": 1.6, "soft_stop_atr": 0.0},
    "soft16m05": {"soft_breakeven_trigger_atr": 1.6, "soft_stop_atr": -0.5},
    "soft20": {"soft_breakeven_trigger_atr": 2.0, "soft_stop_atr": 0.0},
    "soft16_er30": {"soft_breakeven_trigger_atr": 1.6, "soft_stop_atr": 0.0, "min_efficiency_ratio": 0.30},
    "soft16_er35": {"soft_breakeven_trigger_atr": 1.6, "soft_stop_atr": 0.0, "min_efficiency_ratio": 0.35},
    "soft20_er30": {"soft_breakeven_trigger_atr": 2.0, "soft_stop_atr": 0.0, "min_efficiency_ratio": 0.30},
    # Частичная фиксация: половина на 1.6 ATR, остаток ведём трейлом с breakeven.
    "so16": {"scale_out_atr": 1.6, "scale_out_fraction": 0.5, "scale_out_stop_atr": 0.0},
    "so20": {"scale_out_atr": 2.0, "scale_out_fraction": 0.5, "scale_out_stop_atr": 0.5},
    "so25": {"scale_out_atr": 2.5, "scale_out_fraction": 0.5, "scale_out_stop_atr": 1.0},
    "so16_er30": {"scale_out_atr": 1.6, "scale_out_fraction": 0.5, "min_efficiency_ratio": 0.30},
    "so20_er30": {"scale_out_atr": 2.0, "scale_out_fraction": 0.5, "scale_out_stop_atr": 0.5,
                  "min_efficiency_ratio": 0.30},
}

MONTHS = ("2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10")


def _params(overrides: Dict[str, Any], base_variant: str = "run") -> BreakoutParams:
    params = build_params(30, "bar_close", True, base_variant)
    return params.overrides(**overrides) if overrides else params


def _isolated(
    basket: Sequence[str],
    params: BreakoutParams,
    capital: float,
    risk: float,
    candles: Dict[str, Dict[str, Sequence[Any]]],
) -> Dict[str, Any]:
    share = capital / max(len(basket), 1)
    settings = BreakoutSettings(
        capital_rub=share, risk_pct_per_trade=risk, tariff=Tariff.PREMIUM
    )
    results = []
    for ticker in basket:
        series = candles.get(ticker)
        if not series:
            continue
        results.append(
            BreakoutBacktester(load_meta(ticker), params, settings).run(series)
        )
    combined = aggregate(results, capital)
    curve = portfolio_curve(results, capital)
    return {"portfolio": combined, "curve": curve, "results": results}


def _shared(
    basket: Sequence[str],
    params: BreakoutParams,
    capital: float,
    risk: float,
    utilization: float,
    candles: Dict[str, Dict[str, Sequence[Any]]],
) -> Dict[str, Any]:
    instruments = [(load_meta(ticker), params) for ticker in basket]
    engine = SwingAccountBacktester(
        instruments,
        SwingAccountSettings(
            capital_rub=capital,
            risk_pct_per_trade=risk,
            max_margin_utilization_pct=utilization,
        ),
    )
    result = engine.run({ticker: candles[ticker] for ticker in basket if candles.get(ticker)})
    curve = portfolio_curve(list(result.per_ticker.values()), capital)
    profits = sum(t.net_rub for t in result.trades if t.net_rub > 0)
    losses = -sum(t.net_rub for t in result.trades if t.net_rub <= 0)
    pf = None if losses <= 0 else (float("inf") if profits > 0 else 0.0) if profits <= 0 else profits / losses
    portfolio = {
        "trades": len(result.trades),
        "netRub": round(result.net_rub, 2),
        "feesRub": round(result.fees_rub, 2),
        "profitFactor": None if pf is None else round(pf, 3),
        "maxMarginUsed": round(result.max_margin_used, 0),
        "rejectedByMargin": result.rejected_by_margin,
        "rejectedByRisk": result.rejected_by_risk,
    }
    return {"portfolio": portfolio, "curve": curve, "results": result}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Абляция v3.2 (фильтр пилы + мягкий стоп)")
    parser.add_argument("--basket", nargs="*", default=list(DEFAULT_BASKET))
    parser.add_argument("--mode", default="isolated", choices=["isolated", "shared"])
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--risk", type=float, default=1.5)
    parser.add_argument("--utilization", type=float, default=60.0)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--out", default="reports/forts_v32_ablation.json")
    args = parser.parse_args(argv)

    candles = {
        period: {
            ticker: load_candles(ticker, *PERIODS[period], 30) for ticker in args.basket
        }
        for period in ("oos", "is", "full")
    }
    payload: Dict[str, Any] = {
        "mode": args.mode,
        "capital": args.capital,
        "riskPct": args.risk,
        "utilizationPct": args.utilization,
        "basket": list(args.basket),
        "barMinutes": 30,
        "variants": {},
    }
    header = (
        f"{'вариант':<14} {'OOS ₽':>8} {'IS ₽':>8} {'Full ₽':>9} {'PF':>6} {'Sh':>6} "
        f"{'DD%':>6} {'сдел':>5}  "
        + " ".join(f"{month[5:]:>7}" for month in MONTHS)
        + "   jun-sep"
    )
    print(header)
    for name in args.variants:
        params = _params(VARIANTS[name])
        rows = {}
        for period in ("oos", "is", "full"):
            if args.mode == "isolated":
                rows[period] = _isolated(args.basket, params, args.capital, args.risk, candles[period])
            else:
                rows[period] = _shared(
                    args.basket, params, args.capital, args.risk, args.utilization, candles[period]
                )
        full = rows["full"]
        curve = full["curve"]
        portfolio = full["portfolio"]
        monthly = curve.get("monthly", {})
        summer = sum(monthly.get(month, 0.0) for month in ("2026-06", "2026-07", "2026-08", "2026-09"))
        pf = portfolio.get("profitFactor")
        print(
            f"{name:<14} {portfolio_net(rows['oos']):>8.0f} {portfolio_net(rows['is']):>8.0f} "
            f"{portfolio_net(full):>9.0f} "
            f"{('∞' if pf == float('inf') else (f'{pf:.2f}' if pf else '—')):>6} "
            f"{str(curve['sharpe']):>6} {curve['maxDrawdownPct']:>6.2f} {portfolio['trades']:>5}  "
            + " ".join(f"{monthly.get(month, 0.0):>7.0f}" for month in MONTHS)
            + f"  {summer:>8.0f}"
        )
        payload["variants"][name] = {
            "overrides": VARIANTS[name],
            "periods": {
                period: {
                    "portfolio": rows[period]["portfolio"],
                    "curve": rows[period]["curve"],
                }
                for period in rows
            },
        }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {out}")
    return 0


def portfolio_net(row: Dict[str, Any]) -> float:
    return float(row["portfolio"]["netRub"])


if __name__ == "__main__":
    raise SystemExit(main())
