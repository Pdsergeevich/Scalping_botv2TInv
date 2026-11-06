"""Роллинг walk-forward портфель v4.2: ротация корзины каждые K недель.

Скрипт сравнивает статичную корзину v4.1 (отбор один раз на `25.04–25.07`) с
роллинг-ротацией: на каждую дату `T_k` состав `(инструмент, движки)` пересобирается
по трейлинг-окну `L` дней (обе половины окна в плюсе, `PF ≥ 1.30`, `≥ 4` сделок,
дедупликация по базовому активу) и торгуется следующий срез `K` дней.

Пример::

    py -3 scripts/walkforward_portfolio.py --bars 30
    py -3 scripts/walkforward_portfolio.py --bars 30 --conviction
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bot_a.backtest.walkforward import (  # noqa: E402
    WalkForwardSettings,
    build_all_signals,
    run_walkforward,
    save_report,
)
from bot_a.utils.time_provider import MSK  # noqa: E402
from breakout_scan import load_candles, load_meta  # noqa: E402
from multiregime_scan import PERPETUALS, available_continuous, v32_params  # noqa: E402

PERIODS = {
    "oos": ("2026-04-25", "2026-07-25"),
    "is": ("2026-07-25", "2026-11-05"),
    "full": ("2026-04-25", "2026-11-05"),
}


def _ms(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=MSK).timestamp() * 1000)


def universe() -> List[str]:
    cont = available_continuous()
    tickers = [t for t in PERPETUALS if any((ROOT / "data_cache").glob(f"iss_forts_{t}_*.json"))]
    tickers += sorted(cont.values())
    return sorted(set(tickers))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Роллинг walk-forward портфель v4.2")
    parser.add_argument("--bars", type=int, default=30)
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    started = time.time()
    tickers = universe()
    print(f"универсум: {len(tickers)} инструментов", flush=True)
    candles = {ticker: load_candles(ticker, "2026-04-25", "2026-11-05", args.bars) for ticker in tickers}
    metas = {ticker: load_meta(ticker) for ticker in tickers}
    params = {ticker: v32_params(args.bars, True) for ticker in tickers}

    def progress(ticker: str) -> None:
        print(f"  сигналы {ticker}: готово", flush=True)

    signals = build_all_signals(metas, params, candles, progress=progress)
    total = sum(len(series) for series in signals.values())
    print(f"пар (инструмент×движки): {len(signals)}, сделок в потоках: {total}", flush=True)

    start_ms = _ms("2026-04-25")
    end_ms = _ms("2026-11-05")
    oos_end = _ms("2026-07-25")

    cases: List[Dict[str, Any]] = [
        {"name": "v4.1-style step30/L60/top8/r0.7", "step": 30, "lookback": 60, "top": 8,
         "risk": 0.7, "slots": 3, "conviction": False, "min_pf": 1.30, "min_trades": 4},
        {"name": "biweekly L45/top10/r0.7+conv", "step": 14, "lookback": 45, "top": 10,
         "risk": 0.7, "slots": 4, "conviction": True, "min_pf": 1.15, "min_trades": 3},
        {"name": "weekly L45/top10/r0.7+conv", "step": 7, "lookback": 45, "top": 10,
         "risk": 0.7, "slots": 4, "conviction": True, "min_pf": 1.15, "min_trades": 3},
        {"name": "biweekly L45/top12/r0.7+conv", "step": 14, "lookback": 45, "top": 12,
         "risk": 0.7, "slots": 4, "conviction": True, "min_pf": 1.15, "min_trades": 3},
        {"name": "biweekly L30/top10/r0.7+conv", "step": 14, "lookback": 30, "top": 10,
         "risk": 0.7, "slots": 4, "conviction": True, "min_pf": 1.15, "min_trades": 3},
        {"name": "biweekly L45/top10/r1.0+conv", "step": 14, "lookback": 45, "top": 10,
         "risk": 1.0, "slots": 4, "conviction": True, "min_pf": 1.15, "min_trades": 3},
        {"name": "biweekly L45/top10/r1.0 no conv", "step": 14, "lookback": 45, "top": 10,
         "risk": 1.0, "slots": 4, "conviction": False, "min_pf": 1.15, "min_trades": 3},
    ]

    report: Dict[str, Any] = {"bars": args.bars, "capital": args.capital, "cases": {}}
    print(
        f"\n{'случай':<28} {'сдел':>5} {'сдел/нед':>8} {'1-й OOS':>9} {'IS':>9} {'Full':>9} "
        f"{'PF':>6} {'Sh':>6} {'DD%':>6} {'год%':>7}",
        flush=True,
    )
    for case in cases:
        settings = WalkForwardSettings(
            capital_rub=args.capital,
            risk_pct_per_trade=case["risk"],
            step_days=case["step"],
            lookback_days=case["lookback"],
            top=case["top"],
            min_pf=case.get("min_pf", 1.30),
            min_trades=case.get("min_trades", 4),
            max_concurrent_positions=case["slots"],
            conviction_sizing=case["conviction"],
        )
        result = run_walkforward(signals, metas, start_ms, end_ms, settings)
        metrics = result.metrics(args.capital)
        # Сплит по датам выхода: закрытые до 25.07 — «OOS», после — «IS».
        oos_trades = [trade for trade in result.trades if trade.exit_ms < oos_end]
        is_trades = [trade for trade in result.trades if trade.exit_ms >= oos_end]
        oos_net = sum(trade.net_rub for trade in oos_trades)
        is_net = sum(trade.net_rub for trade in is_trades)
        weeks = (end_ms - start_ms) / (7 * 86_400_000)
        report["cases"][case["name"]] = {
            "params": case,
            "metrics": metrics,
            "oosNet": round(oos_net, 2),
            "isNet": round(is_net, 2),
            "baskets": result.baskets,
        }
        print(
            f"{case['name']:<28} {metrics['trades']:>5} {metrics['trades'] / weeks:>8.1f} "
            f"{oos_net:>9.0f} {is_net:>9.0f} {metrics['netRub']:>9.0f} "
            f"{str(metrics['profitFactor']):>6} {str(metrics['sharpe']):>6} "
            f"{metrics['maxDrawdownPct']:>6} {metrics['annualizedPct']:>7.1f}  "
            f"отказы: слоты {result.rejected_by_slots}, риск {result.rejected_by_risk}, "
            f"ГО {result.rejected_by_margin}",
            flush=True,
        )
        print(
            "    месяцы " + " ".join(f"{k[5:]}:{v:.0f}" for k, v in metrics["monthly"].items()),
            flush=True,
        )

    # Эталон: статичная корзина v4.1, но метрики — по фактической equity счёта.
    selection_path = ROOT / f"reports/forts_v40_selection_{args.bars}m.json"
    if selection_path.exists():
        from bot_a.backtest.swing_portfolio import SwingAccountBacktester, SwingAccountSettings
        from multiregime_scan import params_for

        selection = json.loads(selection_path.read_text(encoding="utf-8"))["selection"]
        for period, (date_from, date_to) in PERIODS.items():
            stage = {
                ticker: load_candles(ticker, date_from, date_to, args.bars)
                for ticker in selection
                if ticker in candles
            }
            engine = SwingAccountBacktester(
                [(metas[t], params_for(t, params[t], selection[t])) for t in stage],
                SwingAccountSettings(
                    capital_rub=args.capital,
                    risk_pct_per_trade=0.7,
                    max_margin_utilization_pct=75.0,
                    max_concurrent_positions=3,
                    drawdown_throttle=True,
                    throttle_start_pct=5.0,
                    conviction_sizing=True,
                ),
            )
            outcome = engine.run(stage)
            stats = outcome.metrics(args.capital)
            report.setdefault("static_v41", {})[period] = stats
            print(
                f"static v4.1 [{period}] сделок {stats['trades']:>4} net {stats['netRub']:>9.0f} "
                f"PF {stats['profitFactor']} Sharpe {stats['sharpe']} DD {stats['maxDrawdownPct']}% "
                f"год {stats['annualizedPct']}%",
                flush=True,
            )

    out = Path(args.out or f"reports/forts_v42_walkforward_{args.bars}m.json")
    save_report(out, report)
    print(f"\nсохранено {out} за {time.time() - started:.0f} c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
