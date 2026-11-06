"""Абляция v4.0: универсум × движки (пробой/откат/фейд) × риск-тиры.

Скрипт сравнивает на одних и тех же данных (`data_cache/`, свечи MOEX ISS):

1. `v32` — базовая линия v3.2 (`CNYRUBF + GLDRUBF`, только пробой, ER 0.25);
2. `A` — тот же пробой, но на расширенном универсуме FORTS;
3. `B` — + Engine B (откат к EMA20 в тренде);
4. `C` — + Engine C (фейд растяжения за полосы Кельтнера);
5. `v40` — все три движка, риск-тиры `1.0/1.5/2.0/2.5 %` и утилизация ГО `75/85 %`.

Считает OOS (`25.04–25.07`) / IS (`25.07–05.11`) / Full и месячную разбивку.

Пример::

    py -3 scripts/multiregime_scan.py --bars 30 --max-instruments 12
    py -3 scripts/multiregime_scan.py --bars 15 --basket fx
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bot_a.backtest.swing_portfolio import (  # noqa: E402
    SwingAccountBacktester,
    SwingAccountSettings,
)
from bot_a.backtest.breakout_backtest import portfolio_curve  # noqa: E402
from bot_a.engine.breakout_fsm import BreakoutParams  # noqa: E402
from breakout_scan import CACHE_DIR, load_candles, load_meta  # noqa: E402
from breakout_swing_scan import PERIODS, build_params  # noqa: E402

PERPETUALS = ("CNYRUBF", "GLDRUBF", "USDRUBF", "EURRUBF", "IMOEXF", "SBERF", "GAZPF", "SLVRUBF")
# Ядро FX+металлы+энергия: то, что в v3.1/v3.2 давало положительный результат.
CORE_FAMILIES = ("NG", "BRM", "SILV", "GOLDM", "UCNY")
# Индекс/акции — кандидаты под Engine C (фейд), на пробое исторически убыточны.
EQUITY_FAMILIES = ("MXI", "SBRF", "GAZR", "LKOH", "VTBR", "ROSN", "YDEX", "OZON", "PLZL", "RGBI")


def available_continuous() -> Dict[str, str]:
    """Семейство → тикер continuous, если свечи уже склеены."""
    found: Dict[str, str] = {}
    for path in CACHE_DIR.glob("iss_forts_*_CONT_*.json"):
        family = path.stem.split("_")[2].replace("_CONT", "")
        found[family] = f"{family}_CONT"
    return found


def universe(max_instruments: int, basket: str) -> List[str]:
    cont = available_continuous()
    tickers: List[str] = []
    if basket in ("fx", "all"):
        tickers.extend([t for t in PERPETUALS if (CACHE_DIR / f"iss_forts_{t}_20260901.json").exists() or any(CACHE_DIR.glob(f"iss_forts_{t}_*.json"))])
    if basket in ("core", "all"):
        tickers.extend(cont[f] for f in CORE_FAMILIES if f in cont)
    if basket == "all":
        tickers.extend(cont[f] for f in EQUITY_FAMILIES if f in cont)
    # Детерминированный порядок и отсечение по ликвидности (по числу дней в кэше).
    def days(ticker: str) -> int:
        return len(list(CACHE_DIR.glob(f"iss_forts_{ticker}_*.json")))

    unique = sorted({t for t in tickers})
    unique.sort(key=lambda t: -days(t))
    return unique[:max_instruments]


def v32_params(bars: int, bounce: bool) -> BreakoutParams:
    """База v3.2 (ER 0.25, ночной перенос) + отключённый анти-дожим."""
    base = build_params(bars, "bar_close", bounce, "run")
    return base.overrides(min_efficiency_ratio=0.25)


def with_engines(
    params: BreakoutParams,
    *,
    pullback: bool = False,
    fade: bool = False,
    donchian: int = 0,
) -> BreakoutParams:
    updates: Dict[str, Any] = {"regime_mode": "multi" if (pullback or fade) else "breakout"}
    if pullback:
        updates.update(
            {"pullback_enabled": True, "pullback_min_er": 0.22, "pullback_stop_atr": 1.5,
             "pullback_trail_atr": 3.2}
        )
    if fade:
        updates.update(
            {"fade_enabled": True, "fade_max_er": 0.20, "fade_keltner_atr": 2.2,
             "fade_stop_atr": 1.5, "fade_min_target_fee_ratio": 4.0}
        )
    if donchian:
        updates["donchian_bars"] = donchian
    return params.overrides(**updates)


RANGE_TICKERS = {
    "IMOEXF", "MXI_CONT", "SBRF_CONT", "GAZR_CONT", "LKOH_CONT", "VTBR_CONT",
    "ROSN_CONT", "YDEX_CONT", "OZON_CONT", "PLZLM_CONT", "RGBI_CONT",
    "NASD_CONT", "SPYF_CONT", "ED_CONT", "IBIT_CONT",
}


def params_for(ticker: str, base: BreakoutParams, engines: str) -> BreakoutParams:
    """Набор движков по классу инструмента (без подгонки под конкретный тикер)."""
    if engines == "A":
        return with_engines(base, donchian=16)
    if engines == "AB":
        return with_engines(base, pullback=True, donchian=0 if ticker in RANGE_TICKERS else 16)
    if engines == "AC":
        return with_engines(base, fade=(ticker in RANGE_TICKERS), pullback=(ticker not in RANGE_TICKERS))
    if engines == "ABC":
        return with_engines(base, pullback=True, fade=True)
    if engines == "CLASS":
        if ticker in RANGE_TICKERS:
            return with_engines(base, fade=True)
        return with_engines(base, pullback=True, donchian=16)
    return base


def run_config(
    tickers: Sequence[str],
    base: BreakoutParams,
    engines: str,
    capital: float,
    risk: float,
    utilization: float,
    candles: Dict[str, Dict[str, Sequence[Any]]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for period in ("oos", "is", "full"):
        engine = SwingAccountBacktester(
            [(load_meta(ticker), params_for(ticker, base, engines)) for ticker in tickers],
            SwingAccountSettings(
                capital_rub=capital,
                risk_pct_per_trade=risk,
                max_margin_utilization_pct=utilization,
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
            "maxMargin": round(result.max_margin_used, 0),
            "rejectedMargin": result.rejected_by_margin,
            "rejectedRisk": result.rejected_by_risk,
            "perTicker": {t: round(r.net_rub, 2) for t, r in result.per_ticker.items()},
            "perTickerTrades": {t: len(r.trades) for t, r in result.per_ticker.items()},
            "fees": round(result.fees_rub, 2),
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Мульти-режимная абляция v4.0")
    parser.add_argument("--bars", type=int, default=30)
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--basket", default="all", choices=["fx", "core", "all"])
    parser.add_argument("--max-instruments", type=int, default=12)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    tickers = universe(args.max_instruments, args.basket)
    print(f"универсум ({len(tickers)}): {', '.join(tickers)}", flush=True)
    started = time.time()
    candles: Dict[str, Dict[str, Sequence[Any]]] = {}
    for period in ("oos", "is", "full"):
        stage: Dict[str, Sequence[Any]] = {}
        for ticker in tickers:
            stage[ticker] = load_candles(ticker, *PERIODS[period], args.bars)
        candles[period] = stage
        print(f"  данные {period}: {sum(len(v) for v in stage.values())} баров", flush=True)

    base = v32_params(args.bars, True)
    configs: List[Dict[str, Any]] = [
        {"name": "v32 (2 тикера, пробой)", "tickers": ["CNYRUBF", "GLDRUBF"], "engines": "base",
         "risk": 1.0, "util": 60.0},
        {"name": "A расширенный универсум", "tickers": tickers, "engines": "A",
         "risk": 1.0, "util": 60.0},
        {"name": "A+B откаты (трендовые)", "tickers": tickers, "engines": "AB",
         "risk": 1.0, "util": 60.0},
        {"name": "A+C фейды (индекс/акции)", "tickers": tickers, "engines": "AC",
         "risk": 1.0, "util": 60.0},
        {"name": "v40 по классам", "tickers": tickers, "engines": "CLASS",
         "risk": 1.0, "util": 60.0},
    ]
    for risk, util in ((1.5, 75.0), (2.0, 75.0), (2.5, 85.0)):
        configs.append(
            {
                "name": f"v40 кл. риск {risk:.1f}% / ГО {util:.0f}%",
                "tickers": tickers,
                "engines": "CLASS",
                "risk": risk,
                "util": util,
            }
        )

    report: Dict[str, Any] = {"bars": args.bars, "capital": args.capital, "tickers": tickers, "configs": {}}
    print(
        f"\n{'конфигурация':<26} {'сдел':>5} {'OOS':>9} {'IS':>9} {'Full':>9} "
        f"{'PF':>6} {'Sh':>6} {'DD%':>6} {'год%':>7}",
        flush=True,
    )
    for config in configs:
        data = run_config(
            config["tickers"], base, config["engines"], args.capital, config["risk"],
            config["util"], candles,
        )
        report["configs"][config["name"]] = {
            "risk": config["risk"],
            "utilization": config["util"],
            "engines": config["engines"],
            "tickers": list(config["tickers"]),
            "periods": data,
        }
        full = data["full"]
        weeks = 194 / 7.0
        annual = full["net"] / args.capital * 100.0 * 365.0 / 194.0
        print(
            f"{config['name']:<26} {full['trades']:>5} {data['oos']['net']:>9.0f} "
            f"{data['is']['net']:>9.0f} {full['net']:>9.0f} {str(full['pf']):>6} "
            f"{str(full['sharpe']):>6} {full['dd']:>6} {annual:>7.1f}   "
            f"({full['trades'] / weeks:.1f} сдел/нед, ГО {full['maxMargin']:.0f} ₽)",
            flush=True,
        )
        print("    месяцы " + " ".join(f"{k[5:]}:{v:.0f}" for k, v in full["monthly"].items()), flush=True)

    out = Path(args.out or f"reports/forts_v40_scan_{args.bars}m.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено {out} за {time.time() - started:.0f} c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
