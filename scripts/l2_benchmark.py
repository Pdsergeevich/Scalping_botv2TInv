"""Бенчмарк на РЕАЛЬНЫХ записях L2 (`recordings/`) — v2.7.

Прогоняет `TickReplayBacktester` по записям каждодневного L2 и собирает портфель:
net, PF, DD, сделки/день. Дополнительно умеет применять кластерный фильтр
(«не больше N однонаправленных позиций на общую Leg2») и сравнивать наборы
параметров `v2.6.5` (до патча) и `v2.7`.

Примеры::

    py -3 scripts/l2_benchmark.py --variant v27 --tariff Premium
    py -3 scripts/l2_benchmark.py --variant v265 --tariff Trader --cluster 0
    py -3 scripts/l2_benchmark.py --days 2026-11-02 --variant v27
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot_a.backtest.tick_replay import JsonlRecordingSource, TickReplayBacktester  # noqa: E402
from bot_a.config import AppConfig  # noqa: E402
from bot_a.economics.fees import Tariff  # noqa: E402
from bot_a.risk.cluster import cluster_filter_trades  # noqa: E402

DEFAULT_DAYS = (
    "2026-10-28",
    "2026-10-29",
    "2026-10-30",
    "2026-11-01",
    "2026-11-02",
    "2026-11-03",
)
DEFAULT_PAIRS = ("VTBR_SBER", "MTLR_SBER", "AFKS_SBER", "IRAO_SBER")
#: Значения v2.6.5 (до патча v2.7) — для честного «до/после» на одной и той же записи.
V265_OVERRIDES: Dict[str, Any] = {
    "stat_freeze_until_msk": "",
    "stat_freeze_windows": "",
    "max_valid_spread_bps": 0.0,
    "tp_max_sigma_multiple": 0.0,
    "tp_max_notional_fraction": 0.0,
    "use_volatility_adaptive_stop": False,
    "single_leg_stop_to_target_ratio": 1.5,
    "stop_sigma_multiple": 0.0,
    "use_anti_knife_filter": False,
    "use_stagnation_exit": False,
}
V265_TP_DEV = {"AFKS_SBER": 4.0}
#: v2.7 — всё как в конфиге, но стоп ещё привязан к (урезанной) цели.
V27_OVERRIDES: Dict[str, Any] = {
    "stop_sigma_multiple": 0.0,
    "use_anti_knife_filter": False,
    "use_stagnation_exit": False,
}
#: v2.8 — потолок тейка для ВСЕХ пар (2.2σ / 0.45 % номинала), без выхода из мёртвого спреда.
V28_OVERRIDES: Dict[str, Any] = {
    "tp_max_sigma_multiple": 2.2,
    "tp_max_notional_fraction": 0.0045,
    "tp_cap_beta_threshold": 0.0,
    "use_stagnation_exit": False,
    "use_anti_knife_filter": False,
    "stop_sigma_multiple": 3.5,
}
#: v2.8 + снятый потолок тейка для пар с β ≥ 0.8 (VTBR, MTLR, IRAO).
V28_UNCAPPED_OVERRIDES: Dict[str, Any] = {
    "tp_max_sigma_multiple": 4.5,
    "tp_max_notional_fraction": 0.015,
    "tp_cap_beta_threshold": 0.8,
    "use_stagnation_exit": False,
    "use_anti_knife_filter": False,
    "stop_sigma_multiple": 3.5,
}
#: v2.9 = v2.8 без потолка тейка + выход из мёртвого спреда с защитой от выхода у MAE.
V29_OVERRIDES: Dict[str, Any] = {
    **V28_UNCAPPED_OVERRIDES,
    "use_stagnation_exit": True,
    "stagnation_after_seconds": 780,
    "stagnation_mfe_fee_multiple": 1.0,
    "stagnation_loss_fee_multiple": 1.5,
    "stagnation_min_z_reversion": 0.25,
    "stagnation_max_mae_fraction": 0.7,
}
#: v2.9 без защиты по MAE — ровно как в ТЗ (на реальных L2 ухудшает IRAO на 6.8 RUB).
V29_RAW_OVERRIDES: Dict[str, Any] = {
    **V29_OVERRIDES,
    "stagnation_max_mae_fraction": 0.0,
}
#: v2.10 = v2.9 + анти-нож фильтр по импульсу Leg1.
V210_OVERRIDES: Dict[str, Any] = {
    **V29_OVERRIDES,
    "use_anti_knife_filter": True,
    "anti_knife_window_seconds": 60,
    "anti_knife_baseline_seconds": 300,
    "anti_knife_min_steps": 10,
    "anti_knife_accel_multiple": 1.5,
    "anti_knife_cooldown_seconds": 600,
}
VARIANT_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "v265": V265_OVERRIDES,
    "v27": V27_OVERRIDES,
    "v28": V28_OVERRIDES,
    "v28u": V28_UNCAPPED_OVERRIDES,
    "v29": V29_OVERRIDES,
    "v29raw": V29_RAW_OVERRIDES,
    "v210": V210_OVERRIDES,
}


def _params_for(
    config: AppConfig,
    preset_name: str,
    variant: str,
    unit_lots: bool = False,
    extra: Optional[Dict[str, Any]] = None,
):
    """Параметры пары: персональные настройки + правки выбранного варианта."""
    preset = config.preset(preset_name)
    base = replace(
        config,
        presets={preset_name: preset},
        active_preset=preset_name,
    ).with_preset_overrides()
    params = base.strategy
    if unit_lots:
        params = replace(params, quantity_lots=1)
    variant_overrides = VARIANT_OVERRIDES.get(variant, {})
    if variant_overrides:
        params = replace(params, **variant_overrides)
    if variant == "v265" and preset_name in V265_TP_DEV:
        params = replace(params, tp_deviation_fraction=V265_TP_DEV[preset_name])
    if extra:
        valid = {key: value for key, value in extra.items() if hasattr(params, key)}
        if valid:
            params = replace(params, **valid)
    return base, preset, params


def _run_pair(
    config: AppConfig,
    preset_name: str,
    days: Sequence[str],
    variant: str,
    recordings_dir: Path,
    unit_lots: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Прогоняет все дни одной пары, возвращает сделки и дневные итоги."""
    cfg, preset, params = _params_for(config, preset_name, variant, unit_lots, extra)
    day_rows: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    for day in days:
        path = recordings_dir / f"{preset_name}_{day}.jsonl.gz"
        if not path.exists() or path.stat().st_size == 0:
            day_rows.append({"date": day, "net": 0.0, "trades": 0, "missing": True})
            continue
        source = JsonlRecordingSource.from_path(path)
        backtester = TickReplayBacktester(cfg, preset, source, params=params)
        try:
            result = asyncio.run(backtester.run(write_report=False))
        except Exception as exc:  # noqa: BLE001 — нет событий/битый файл
            day_rows.append(
                {"date": day, "net": 0.0, "trades": 0, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        day_rows.append(
            {
                "date": day,
                "net": float(result.totals.net_pnl_rub),
                "gross": float(result.totals.gross_pnl_rub),
                "fees": float(result.totals.fees_rub),
                "trades": int(result.totals.trades),
                "events": int(result.events_processed),
            }
        )
        for trade in result.trades:
            trades.append(
                {
                    "open_ms": int(trade.opened_ms),
                    "close_ms": int(trade.closed_ms),
                    "side": trade.side_leg1.value.upper(),
                    "net": float(trade.net_pnl_rub),
                    "gross": float(trade.gross_pnl_rub),
                    "fees": float(trade.fees_rub),
                    "reason": trade.exit_reason.value,
                    "day": day,
                }
            )
    return {"pair": preset_name, "days": day_rows, "trades": trades}


def _portfolio(
    runs: Sequence[Dict[str, Any]],
    cluster: int,
    leg2_by_pair: Dict[str, str],
) -> Dict[str, Any]:
    """Собирает портфель: опционально гасит коррелированные входы кластера."""
    trades_by_pair = {run["pair"]: run["trades"] for run in runs}
    kept, skipped = cluster_filter_trades(trades_by_pair, leg2_by_pair, max_same_direction=cluster)
    per_pair: Dict[str, Any] = {}
    daily: Dict[str, float] = {}
    all_net: List[float] = []
    wins = losses = 0.0
    trades_total = 0
    for pair, items in kept.items():
        net = sum(item["net"] for item in items)
        per_pair[pair] = {
            "trades": len(items),
            "net": round(net, 4),
            "dropped": len(trades_by_pair[pair]) - len(items),
        }
        trades_total += len(items)
        for item in items:
            all_net.append(item["net"])
            daily[item["day"]] = daily.get(item["day"], 0.0) + item["net"]
            if item["net"] >= 0:
                wins += item["net"]
            else:
                losses += -item["net"]
    for run in runs:  # дни без сделок тоже должны попасть в эквити
        for row in run["days"]:
            daily.setdefault(row["date"], 0.0)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for day in sorted(daily):
        equity += daily[day]
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    sharpe = 0.0
    values = list(daily.values())
    if len(values) > 1:
        mean = statistics.fmean(values)
        std = statistics.pstdev(values)
        if std > 1e-9:
            sharpe = mean / std * (len(values) ** 0.5)
    return {
        "cluster": cluster,
        "trades": trades_total,
        "tradesPerDay": round(trades_total / max(len(daily), 1), 2),
        "net": round(sum(all_net), 4),
        "profitFactor": round(wins / losses, 3) if losses > 1e-9 else None,
        "maxDrawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 2),
        "clusterSkips": skipped,
        "perPair": per_pair,
        "daily": {day: round(value, 4) for day, value in sorted(daily.items())},
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Бенчмарк реальных L2-записей")
    parser.add_argument("--days", nargs="*", default=list(DEFAULT_DAYS))
    parser.add_argument("--pairs", nargs="*", default=list(DEFAULT_PAIRS))
    parser.add_argument(
        "--variant", choices=list(VARIANT_OVERRIDES), default="v29"
    )
    parser.add_argument("--tariff", default="Premium", choices=["Trader", "Premium"])
    parser.add_argument("--cluster", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--recordings", default=str(ROOT / "recordings"))
    parser.add_argument(
        "--stop-sigma",
        dest="stop_sigma",
        type=float,
        default=3.5,
        help="stopSigmaMultiple для вариантов v28/v28u/v29 (0 — поведение v2.7)",
    )
    parser.add_argument(
        "--knife-steps",
        dest="knife_steps",
        type=int,
        default=0,
        help="переопределить antiKnifeMinSteps (0 — из варианта)",
    )
    parser.add_argument(
        "--knife-accel",
        dest="knife_accel",
        type=float,
        default=0.0,
        help="переопределить antiKnifeAccelMultiple (0 — из варианта)",
    )
    parser.add_argument(
        "--knife-cooldown",
        dest="knife_cooldown",
        type=int,
        default=-1,
        help="переопределить antiKnifeCooldownSeconds (отрицательное — из варианта)",
    )
    parser.add_argument(
        "--knife-window",
        dest="knife_window",
        type=int,
        default=0,
        help="переопределить antiKnifeWindowSeconds (0 — из варианта)",
    )
    parser.add_argument(
        "--max-hold",
        dest="max_hold",
        type=int,
        default=0,
        help="переопределить maxHoldSeconds (0 — из конфига)",
    )
    parser.add_argument(
        "--stagnation-seconds",
        dest="stagnation_seconds",
        type=int,
        default=0,
        help="переопределить stagnationAfterSeconds (0 — из варианта)",
    )
    parser.add_argument(
        "--tp-sigma",
        dest="tp_sigma",
        type=float,
        default=-1.0,
        help="переопределить tpMaxSigmaMultiple (отрицательное — из варианта)",
    )
    parser.add_argument(
        "--tp-notional",
        dest="tp_notional",
        type=float,
        default=-1.0,
        help="переопределить tpMaxNotionalFraction (отрицательное — из варианта)",
    )
    parser.add_argument(
        "--tp-cap-beta",
        dest="tp_cap_beta",
        type=float,
        default=-1.0,
        help="переопределить tpCapBetaThreshold (отрицательное — из варианта)",
    )
    parser.add_argument(
        "--stagnation-z",
        dest="stagnation_z",
        type=float,
        default=-1.0,
        help="переопределить stagnationMinZReversion (отрицательное — из варианта)",
    )
    parser.add_argument(
        "--stagnation-mae",
        dest="stagnation_mae",
        type=float,
        default=-1.0,
        help="переопределить stagnationMaxMaeFraction (0.7 — не выходить у худшего уровня)",
    )
    parser.add_argument(
        "--stagnation-mfe",
        dest="stagnation_mfe",
        type=float,
        default=-1.0,
        help="переопределить stagnationMfeFeeMultiple",
    )
    parser.add_argument(
        "--stagnation-loss",
        dest="stagnation_loss",
        type=float,
        default=-1.0,
        help="переопределить stagnationLossFeeMultiple",
    )
    parser.add_argument(
        "--tp-dev",
        dest="tp_dev",
        nargs="*",
        default=[],
        help='переопределение tpDeviationFraction, например AFKS_SBER=0.85',
    )
    parser.add_argument(
        "--stop-ratio",
        dest="stop_ratio",
        type=float,
        default=None,
        help="переопределение singleLegStopToTargetRatio (0 — стоп только от σ и шагов)",
    )
    parser.add_argument(
        "--unit-lots",
        dest="unit_lots",
        action="store_true",
        help="??? ???? ?? 1 ???? (??????????? ? portfolio_sim)",
    )
    parser.add_argument("--json", dest="json_out", default="")
    args = parser.parse_args(argv)

    config = AppConfig.load(str(ROOT / "appsettings.json"))
    config = config.with_overrides(tariff=Tariff(args.tariff))
    recordings_dir = Path(args.recordings)

    runs: List[Dict[str, Any]] = []
    tp_dev_map: Dict[str, float] = {}
    for item in args.tp_dev:
        name, _, value = item.partition("=")
        tp_dev_map[name.strip()] = float(value)
    for pair in args.pairs:
        extra: Dict[str, Any] = {}
        if args.variant in ("v28", "v28u", "v29"):
            extra["stop_sigma_multiple"] = float(args.stop_sigma)
        if args.max_hold > 0:
            extra["max_hold_seconds"] = int(args.max_hold)
        if args.knife_steps > 0:
            extra["anti_knife_min_steps"] = int(args.knife_steps)
        if args.knife_accel > 0:
            extra["anti_knife_accel_multiple"] = float(args.knife_accel)
        if args.knife_cooldown >= 0:
            extra["anti_knife_cooldown_seconds"] = int(args.knife_cooldown)
        if args.knife_window > 0:
            extra["anti_knife_window_seconds"] = int(args.knife_window)
        if args.stagnation_seconds > 0:
            extra["stagnation_after_seconds"] = int(args.stagnation_seconds)
            extra["use_stagnation_exit"] = True
        if args.stagnation_z >= 0:
            extra["stagnation_min_z_reversion"] = float(args.stagnation_z)
        if args.stagnation_mfe >= 0:
            extra["stagnation_mfe_fee_multiple"] = float(args.stagnation_mfe)
        if args.stagnation_mae >= 0:
            extra["stagnation_max_mae_fraction"] = float(args.stagnation_mae)
        if args.stagnation_loss >= 0:
            extra["stagnation_loss_fee_multiple"] = float(args.stagnation_loss)
        if args.tp_sigma >= 0:
            extra["tp_max_sigma_multiple"] = float(args.tp_sigma)
        if args.tp_notional >= 0:
            extra["tp_max_notional_fraction"] = float(args.tp_notional)
        if args.tp_cap_beta >= 0:
            extra["tp_cap_beta_threshold"] = float(args.tp_cap_beta)
        if args.stop_ratio is not None:
            extra["single_leg_stop_to_target_ratio"] = float(args.stop_ratio)
        if pair in tp_dev_map:
            extra["tp_deviation_fraction"] = tp_dev_map[pair]
        run = _run_pair(
            config,
            pair,
            args.days,
            args.variant,
            recordings_dir,
            args.unit_lots,
            extra or None,
        )
        runs.append(run)
        net = sum(row.get("net", 0.0) for row in run["days"])
        trades = sum(row.get("trades", 0) for row in run["days"])
        print(
            f"{pair:<12}{args.variant:<5}{args.tariff:<8}сдел {trades:>3}  "
            f"net {net:>10.3f}  дней {len(run['days'])}"
        )

    leg2_by_pair = {pair: config.preset(pair).leg2.ticker for pair in args.pairs}
    portfolios = [_portfolio(runs, int(cap), leg2_by_pair) for cap in args.cluster]
    print()
    print(f"{'кластер':<9}{'сделок':>8}{'/день':>7}{'NET':>10}{'PF':>7}{'DD':>9}{'Sharpe':>8}{'отсечек':>9}")
    for portfolio in portfolios:
        pf = portfolio["profitFactor"]
        print(
            f"{portfolio['cluster']:<9}{portfolio['trades']:>8}{portfolio['tradesPerDay']:>7.2f}"
            f"{portfolio['net']:>10.3f}{('—' if pf is None else f'{pf:.2f}'):>7}"
            f"{portfolio['maxDrawdown']:>9.2f}{portfolio['sharpe']:>8.2f}{portfolio['clusterSkips']:>9}"
        )

    if args.json_out:
        payload = {
            "variant": args.variant,
            "tariff": args.tariff,
            "stopSigma": args.stop_sigma,
            "stopRatio": args.stop_ratio,
            "stagnationSeconds": args.stagnation_seconds or None,
            "maxHoldSeconds": args.max_hold or None,
            "stagnationMae": args.stagnation_mae if args.stagnation_mae >= 0 else None,
            "tpSigma": args.tp_sigma if args.tp_sigma >= 0 else None,
            "tpNotional": args.tp_notional if args.tp_notional >= 0 else None,
            "tpCapBeta": args.tp_cap_beta if args.tp_cap_beta >= 0 else None,
            "tpDev": tp_dev_map,
            "days": list(args.days),
            "pairs": list(args.pairs),
            "runs": runs,
            "portfolios": portfolios,
        }
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"\nJSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
