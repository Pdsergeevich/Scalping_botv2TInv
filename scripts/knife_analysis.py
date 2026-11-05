"""Анализ «медленного ножа»: 3-минутный дрейф Leg1 против сделки в шагах и σ (v2.12).

Считает по каждому входу из бенчмарка набор величин, которые может использовать фильтр
входа, и печатает их рядом с результатом сделки — чтобы видеть, какие пороги отделяют
убыточные входы от прибыльных.

Пример::

    py -3 scripts/knife_analysis.py --trades reports/l2_bench_v211_final/v211_7d.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot_a.config import AppConfig  # noqa: E402
from bot_a.utils.time_provider import ms_to_msk  # noqa: E402
from scripts.l2_benchmark import VARIANT_OVERRIDES  # noqa: E402
from scripts.lead_diagnostics import Sample, build_series, window_stats  # noqa: E402


def trade_metrics(
    pair_name: str,
    trade: Dict[str, Any],
    series: Sequence[Sample],
    times: Sequence[int],
    step: float,
) -> Dict[str, float]:
    """Все метрики фильтра для одного входа."""
    sign = 1 if trade["side"] == "BUY" else -1
    entry_ms = int(trade["open_ms"])
    one = window_stats(series, times, entry_ms, 60, sign)
    three = window_stats(series, times, entry_ms, 180, sign)
    five = window_stats(series, times, entry_ms, 300, sign)
    if not (one and three and five):
        return {}
    step_rub = step  # шаг цены в рублях за 1 лот (rub_per_price_unit == 1 для TQBR)
    steps_1m = three and one["adverse1"] / step_rub if step_rub > 0 else 0.0
    steps_3m = three["adverse1"] / step_rub if step_rub > 0 else 0.0
    sigma_leg1 = three["sigma_leg1"]
    accel = (
        abs(one["dv1"]) / (abs(five["dv1"]) / 5.0) if abs(five["dv1"]) > 1e-9 else 0.0
    )
    return {
        "net": float(trade["net"]),
        "side": trade["side"],
        "steps_1m": steps_1m,
        "steps_3m": steps_3m,
        "accel": accel,
        "sigma_3m": (three["adverse1"] / sigma_leg1) if sigma_leg1 > 1e-9 else 0.0,
        "lead_share_3m": three["lead_share"],
        "sigma_leg1": sigma_leg1,
        "entry_z": float(trade.get("entry_z", 0.0)),
        "mfe": float(trade.get("mfe", 0.0)),
        "mae": float(trade.get("mae", 0.0)),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Анализ медленного ножа (v2.12)")
    parser.add_argument("--trades", default="reports/l2_bench_v211_final/v211_7d.json")
    parser.add_argument("--variant", default="v211", choices=list(VARIANT_OVERRIDES))
    args = parser.parse_args(argv)
    config = AppConfig.load(str(ROOT / "appsettings.json"))
    payload = json.loads(Path(args.trades).read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    cache: Dict[tuple, List[Sample]] = {}
    for run in payload["runs"]:
        pair_name = run["pair"]
        preset = config.preset(pair_name)
        step = float(preset.leg1.min_price_increment) * float(preset.leg1.rub_per_price_unit)
        for trade in run["trades"]:
            key = (pair_name, trade["day"])
            if key not in cache:
                cache[key] = build_series(pair_name, trade["day"], config, args.variant)
            series = cache[key]
            if not series:
                continue
            metrics = trade_metrics(
                pair_name, trade, series, [item.time_ms for item in series], step
            )
            if not metrics:
                continue
            metrics["label"] = (
                f"{pair_name.split('_')[0]} "
                f"{ms_to_msk(int(trade['open_ms'])):%d.%m %H:%M} {trade['side']}"
            )
            metrics["reason"] = trade["reason"]
            rows.append(metrics)
    header = (
        "| вход | NET ₽ | шагов 1м | шагов 3м | σ за 3м | lead_share 3м | ускорение "
        "| entry Z | MFE ₽ | MAE ₽ | выход |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    print(header)
    for row in sorted(rows, key=lambda item: item["net"]):
        print(
            f"| {row['label']} | {row['net']:+.2f} | {row['steps_1m']:.1f} | "
            f"{row['steps_3m']:.1f} | {row['sigma_3m']:+.2f} | {row['lead_share_3m']:+.2f} | "
            f"{row['accel']:.2f} | {row['entry_z']:+.2f} | {row['mfe']:+.2f} | "
            f"{row['mae']:+.2f} | {row['reason']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
