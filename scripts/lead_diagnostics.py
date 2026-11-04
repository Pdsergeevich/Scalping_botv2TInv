"""Диагностика лид-лаг микроструктуры входов на реальных L2-записях (v2.10).

Отвечает на вопрос «кто именно создал девиацию спреда перед входом»: ведущая нога
(Leg2) на ралли/падении или сама Leg1 идёт против сделки (ловля падающего ножа).
Считает по каждому входу окна 1 / 3 / 5 минут до сигнала:

* ``ΔV1`` / ``ΔV2`` — изменение рублёвой стоимости ног (по микропрайсу);
* ``leg2_part = ΔV2``, ``leg1_part = −β·ΔV1`` — вклад каждой ноги в рост спреда;
* ``lead_share = leg2_part / (|leg2_part| + |leg1_part|)`` в направлении сделки;
* ``slope1`` / ``slope2`` — наклон в рублях за минуту;
* ``obi1`` / ``obi2``, ``beta_drift`` — состояние стакана и дрейф беты.

Пример::

    py -3 scripts/lead_diagnostics.py --trades reports/l2_bench_v29/v29_p3.json
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot_a.backtest.tick_replay import JsonlRecordingSource  # noqa: E402
from bot_a.config import AppConfig  # noqa: E402
from bot_a.math_engine.pair_spread import PairSpreadCalculator  # noqa: E402
from bot_a.utils.time_provider import ms_to_msk  # noqa: E402
from scripts.l2_benchmark import VARIANT_OVERRIDES, _params_for  # noqa: E402

WINDOWS_SEC = (60, 180, 300)


@dataclass(slots=True)
class Sample:
    time_ms: int
    v1: float
    v2: float
    beta: float
    obi1: float
    obi2: float
    z: float
    beta_drift: float
    spread: float
    mean: float
    sigma_spread: float


def build_series(pair: str, day: str, config: AppConfig, variant: str) -> List[Sample]:
    """Прогоняет запись дня через тот же калькулятор, что и ядро, и снимает ряды."""
    preset = config.preset(pair)
    _cfg, _preset, params = _params_for(config, pair, variant)
    calc = PairSpreadCalculator(
        leg1=preset.leg1,
        leg2=preset.leg2,
        beta=params.beta,
        ewma_window=params.ewma_window,
        obi_depth=params.obi_depth,
        beta_mode=params.beta_mode,
        beta_min=params.beta_min,
        beta_max=params.beta_max,
        beta_min_samples=params.beta_min_samples,
        macro_beta_window=params.macro_beta_window,
        macro_beta_min_samples=params.macro_beta_min_samples,
        stat_sample_interval_seconds=params.stat_sample_interval_seconds,
        stat_freeze_until_msk=params.stat_freeze_until_msk,
        stat_freeze_windows=params.stat_freeze_windows,
        max_valid_spread_bps=params.max_valid_spread_bps,
    )
    path = ROOT / "recordings" / f"{pair}_{day}.jsonl.gz"
    if not path.exists():
        return []
    source = JsonlRecordingSource([path])
    series: List[Sample] = []

    async def run() -> None:
        async for event in source.events(0, 2**62):
            if event.is_book and event.book is not None:
                metrics = calc.on_book(event.leg - 1, event.book)
                if metrics is None:
                    continue
                series.append(
                    Sample(
                        time_ms=event.time_ms,
                        v1=float(preset.leg1.value_in_rub(metrics.micro_leg1)),
                        v2=float(preset.leg2.value_in_rub(metrics.micro_leg2)),
                        beta=float(metrics.beta),
                        obi1=float(metrics.obi_leg1),
                        obi2=float(metrics.obi_leg2),
                        z=float(metrics.zscore),
                        beta_drift=float(metrics.beta_drift or 0.0),
                        spread=float(metrics.spread_rub),
                        mean=float(metrics.mean_rub),
                        sigma_spread=float(metrics.variance) ** 0.5,
                    )
                )

    asyncio.run(run())
    return series


def _index_at(times: Sequence[int], target_ms: int) -> int:
    """Индекс последнего сэмпла с ``time <= target`` (``-1``, если такого нет)."""
    return bisect.bisect_right(times, target_ms) - 1


def window_stats(
    series: Sequence[Sample], times: Sequence[int], entry_ms: int, window_sec: int, sign: int
) -> Dict[str, float]:
    """Показатели за ``window_sec`` секунд до входа, приведённые к направлению сделки."""
    now_idx = _index_at(times, entry_ms)
    past_idx = _index_at(times, entry_ms - window_sec * 1000)
    if now_idx < 0 or past_idx < 0 or now_idx == past_idx:
        return {}
    now, past = series[now_idx], series[past_idx]
    beta = now.beta if abs(now.beta) > 1e-9 else 1.0
    dv1 = now.v1 - past.v1
    dv2 = now.v2 - past.v2
    # Вклад ног в рост спреда S = V2 − β·V1.
    leg2_part = dv2
    leg1_part = -beta * dv1
    total = abs(leg2_part) + abs(leg1_part)
    lead_share = leg2_part / total if total > 1e-9 else 0.0
    minutes = window_sec / 60.0
    sigma_leg1 = now.sigma_spread / beta if now.sigma_spread > 0 else 0.0
    return {
        "dv1": dv1,
        "dv2": dv2,
        "leg1_part": leg1_part,
        "leg2_part": leg2_part,
        "lead_share": lead_share,
        "slope1": dv1 / minutes,
        "slope2": dv2 / minutes,
        "obi1": now.obi1,
        "obi2": now.obi2,
        "beta": beta,
        "beta_drift": now.beta_drift,
        "z": now.z,
        "sigma_leg1": sigma_leg1,
        # Знак «против сделки»: положительное значение = Leg1 двигалась против входа
        # (падала перед BUY, росла перед SELL) — это и есть «нож».
        "adverse1": -dv1 * sign,
        "adverse1_sigma": (-dv1 * sign / sigma_leg1) if sigma_leg1 > 1e-9 else 0.0,
    }


def trade_row(
    pair_name: str, trade: Dict[str, Any], series: Sequence[Sample], times: Sequence[int]
) -> str:
    """Строка отчёта: окна 1/3/5 минут перед входом + ускорение движения Leg1."""
    sign = 1 if trade["side"] == "BUY" else -1
    entry_ms = int(trade["open_ms"])
    stats = {sec: window_stats(series, times, entry_ms, sec, sign) for sec in WINDOWS_SEC}
    if not all(stats.values()):
        return ""
    one, three, five = stats[60], stats[180], stats[300]
    accel = (
        abs(one["dv1"]) / (abs(five["dv1"]) / 5.0) if abs(five["dv1"]) > 1e-9 else 0.0
    )
    label = f"{pair_name.split('_')[0]} {ms_to_msk(entry_ms):%d.%m %H:%M:%S} {trade['side']}"
    return (
        f"| {label} | {trade['net']:+.2f} | {one['lead_share']:+.2f}/{three['lead_share']:+.2f}"
        f"/{five['lead_share']:+.2f} | {one['dv1']:+.3f}/{three['dv1']:+.3f}/{five['dv1']:+.3f}"
        f" | {one['dv2']:+.3f}/{three['dv2']:+.3f}/{five['dv2']:+.3f}"
        f" | {one['adverse1_sigma']:+.2f}/{three['adverse1_sigma']:+.2f}"
        f" | {accel:.2f} | {one['sigma_leg1']:.3f} | {one['obi1']:+.2f}/{one['obi2']:+.2f} |"
    )


def describe(
    pair: str, trades_path: Path, variant: str, days: Optional[Sequence[str]]
) -> str:
    config = AppConfig.load(str(ROOT / "appsettings.json"))
    payload = json.loads(trades_path.read_text(encoding="utf-8"))
    rows: List[str] = []
    cache: Dict[tuple, List[Sample]] = {}
    for run in payload["runs"]:
        pair_name = run["pair"]
        if pair != "ALL" and pair_name != pair:
            continue
        for trade in run["trades"]:
            day = trade["day"]
            if days and day not in days:
                continue
            key = (pair_name, day)
            if key not in cache:
                cache[key] = build_series(pair_name, day, config, variant)
            series = cache[key]
            if not series:
                continue
            row = trade_row(pair_name, trade, series, [item.time_ms for item in series])
            if row:
                rows.append(row)
    return "\n".join(rows)


HEADER = (
    "| вход | NET ₽ | lead_share 1m/3m/5m | ΔV1 1m/3m/5m | ΔV2 1m/3m/5m "
    "| Leg1 против /σ 1m/3m | ускорение | σ_Leg1 | OBI1/OBI2 |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Лид-лаг диагностика входов (v2.10)")
    parser.add_argument("--trades", default="reports/l2_bench_v29/v29_p3.json")
    parser.add_argument("--pair", default="ALL")
    parser.add_argument("--variant", default="v29", choices=list(VARIANT_OVERRIDES))
    parser.add_argument("--days", nargs="*", default=None)
    args = parser.parse_args(argv)
    print(HEADER)
    print(describe(args.pair, Path(args.trades), args.variant, args.days))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
