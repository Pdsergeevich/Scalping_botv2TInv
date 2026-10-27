"""Калибровка Z: реальный L2 против синтетического свечного моста (v2.5).

Свечной бэктест строит стакан мостом ``open → low → high → close`` и на 5-секундной
каденции выборка попадает в экстремумы внутриминутного хода. Из-за этого хвосты
``Z`` раздуты, и порог ``Z_entry``, подобранный на свечах, на живом стакане
срабатывает реже (26.10.2026: 153 срабатывания в реплее против 0 в live).

Скрипт считает по одному и тому же интервалу времени:
  * ``live``  — Z из живого стакана (трейл ``state/watch_<пресет>_<дата>.jsonl``);
  * ``ticks`` — Z из свечей с синтетическим мостом (``--synth-mode ticks``);
  * ``close`` — Z из свечей только по закрытиям (``--synth-mode close``).

и печатает стандартные отклонения, перцентили ``|Z|`` (p95/p99/p99.5) и их отношения.

Примеры:
    py -3 scripts/l2_calibration.py --date 2026-10-26 --from-time 18:50 --to-time 21:30
    py -3 scripts/l2_calibration.py --date 2026-10-26 --preset VTBR_SBER --json reports/calib.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import replace
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot_a.backtest.historical_candles import iter_candle_events  # noqa: E402
from bot_a.backtest.iss_source import IssCandleSource  # noqa: E402
from bot_a.config import AppConfig  # noqa: E402
from bot_a.economics.fees import TinkoffFeeCalculator  # noqa: E402
from bot_a.engine.factory import create_engine, create_guard  # noqa: E402
from bot_a.engine.ledger import PortfolioLedger  # noqa: E402
from bot_a.execution.simulated_gateway import SimulatedOrderExecutionGateway  # noqa: E402
from bot_a.logging_setup import configure_logging  # noqa: E402
from bot_a.services.bootstrap import bootstrap_warmup  # noqa: E402
from bot_a.utils.time_provider import MSK, SimulatedTimeProvider, msk_to_ms  # noqa: E402


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _stats(series: Sequence[float], z_entry: float) -> Dict[str, Any]:
    absolute = [abs(value) for value in series]
    return {
        "n": len(series),
        "std": round(statistics.pstdev(series), 4) if len(series) > 1 else 0.0,
        "mean": round(statistics.fmean(series), 4) if series else 0.0,
        "absMean": round(statistics.fmean(absolute), 4) if absolute else 0.0,
        "maxAbs": round(max(absolute), 3) if absolute else 0.0,
        "p95": round(_percentile(absolute, 0.95), 3),
        "p99": round(_percentile(absolute, 0.99), 3),
        "p995": round(_percentile(absolute, 0.995), 3),
        "aboveEntry": sum(1 for value in absolute if value >= z_entry),
        "aboveEntryPct": round(
            100.0 * sum(1 for value in absolute if value >= z_entry) / max(len(absolute), 1), 3
        ),
    }


def _load_live_trail(state_dir: Path, preset: str, day: str) -> List[Tuple[int, float]]:
    path = Path(state_dir) / f"watch_{preset}_{day}.jsonl"
    if not path.exists():
        return []
    points: List[Tuple[int, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            stamp = datetime.fromisoformat(row["t"])
        except (ValueError, KeyError):
            continue
        points.append((int(stamp.timestamp() * 1000), float(row["z"])))
    points.sort(key=lambda item: item[0])
    return points


async def _candle_series(
    config: AppConfig,
    preset: Any,
    start_ms: int,
    end_ms: int,
    synth_mode: str,
    warmup_days: int,
    sample_ms: int,
    ewma_window: Optional[int] = None,
    macro_window: Optional[int] = None,
) -> List[Tuple[int, float]]:
    from bot_a.math_engine.pair_spread import PairSpreadCalculator  # noqa: F401 — побочный импорт

    clock = SimulatedTimeProvider(start_ms)
    fee = TinkoffFeeCalculator(config.fees.tariff, clock=clock)
    gateway = SimulatedOrderExecutionGateway(
        metadata=[preset.leg1, preset.leg2],
        clock=clock,
        fee_calculator=fee,
        latency_ms=0,
        queue_fill_factor=config.execution.queue_fill_factor,
        slippage_steps=0,
        order_book_depth=config.execution.order_book_depth,
        initial_cash_rub=config.execution.paper_initial_cash_rub,
    )
    guard = create_guard(config, [preset.leg1, preset.leg2])
    ledger = PortfolioLedger(config.execution.paper_initial_cash_rub)
    params = replace(
        config.strategy,
        beta=float(preset.beta),
        beta_mode="slow_ratio",
        use_entry_persistence=False,
        ewma_window=int(ewma_window or config.strategy.ewma_window),
        macro_beta_window=int(macro_window or config.strategy.macro_beta_window),
    )
    engine = create_engine(
        [preset.leg1, preset.leg2],
        params,
        config.execution.topology,
        gateway,
        fee,
        clock,
        guard,
        ledger,
    )
    await gateway.start()
    source = IssCandleSource(config.path(config.storage.data_cache_dir))
    if warmup_days > 0:
        await bootstrap_warmup(
            config, preset, engine, source=source, days=warmup_days, end_ms=start_ms - 1, clock=clock
        )
    candles1, candles2 = await asyncio.gather(
        source.get_candles(preset.leg1, start_ms, end_ms, "1min"),
        source.get_candles(preset.leg2, start_ms, end_ms, "1min"),
    )
    series: List[Tuple[int, float]] = []
    last_sample = -1
    events = iter_candle_events(
        preset.leg1,
        preset.leg2,
        list(candles1),
        list(candles2),
        depth=config.execution.order_book_depth,
        phase_lock=True,
        synth_mode=synth_mode,
    )
    for event in events:
        clock.set(event.time_ms)
        await engine.on_clock()
        if event.is_book and event.book is not None:
            await engine.on_order_book(event.leg, event.book)
        elif event.trade is not None:
            await engine.on_trade(event.leg, event.trade)
        metrics = engine.metrics
        if metrics is None:
            continue
        if int(metrics.time_ms) - last_sample >= sample_ms:
            last_sample = int(metrics.time_ms)
            series.append((int(metrics.time_ms), float(metrics.zscore)))
    await gateway.stop()
    return series


def _align(
    live: Sequence[Tuple[int, float]], other: Sequence[Tuple[int, float]]
) -> List[Tuple[float, float]]:
    """Сопоставляет точки по ближайшей метке времени в пределах 6 секунд."""
    if not live or not other:
        return []
    pairs: List[Tuple[float, float]] = []
    index = 0
    for stamp, value in live:
        while index + 1 < len(other) and abs(other[index + 1][0] - stamp) <= abs(other[index][0] - stamp):
            index += 1
        if abs(other[index][0] - stamp) <= 6_000:
            pairs.append((value, other[index][1]))
    return pairs


def _window(day: str, value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    date = datetime.fromisoformat(day).date()
    return msk_to_ms(datetime.combine(date, time(hour, minute, tzinfo=MSK)))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Калибровка Z: live L2 vs свечи")
    parser.add_argument("--date", default=datetime.now(tz=MSK).strftime("%Y-%m-%d"))
    parser.add_argument("--preset", default="VTBR_SBER")
    parser.add_argument("--tariff", default="Premium")
    parser.add_argument("--from-time", default="18:50")
    parser.add_argument("--to-time", default="23:00")
    parser.add_argument("--z-entry", type=float, default=None)
    parser.add_argument("--warmup-days", type=int, default=5)
    parser.add_argument(
        "--close-ewma-window",
        type=int,
        default=200,
        help="окно EWMA для режима close (1 замер в минуту вместо 12 подтиков: 2400/12)",
    )
    parser.add_argument(
        "--close-macro-window",
        type=int,
        default=1200,
        help="макро-окно беты для режима close (14400/12)",
    )
    parser.add_argument("--config", default="appsettings.json")
    parser.add_argument("--json", dest="json_out", default="")
    args = parser.parse_args(argv)

    configure_logging("ERROR", "logs", "l2_calibration.log")
    config = AppConfig.load(args.config)
    if args.tariff:
        config = config.with_overrides(tariff=args.tariff)
    config = config.with_overrides(active_preset=args.preset).with_preset_overrides()
    preset = config.preset()
    z_entry = float(args.z_entry if args.z_entry is not None else config.strategy.z_entry)
    start_ms = _window(args.date, args.from_time)
    end_ms = _window(args.date, args.to_time)

    live_all = _load_live_trail(config.path(config.storage.state_dir), args.preset, args.date)
    live = [(stamp, value) for stamp, value in live_all if start_ms <= stamp <= end_ms]
    ticks = asyncio.run(
        _candle_series(config, preset, start_ms, end_ms, "ticks", args.warmup_days, 5_000)
    )
    close = asyncio.run(
        _candle_series(
            config,
            preset,
            start_ms,
            end_ms,
            "close",
            args.warmup_days,
            5_000,
            ewma_window=args.close_ewma_window,
            macro_window=args.close_macro_window,
        )
    )

    aligned_ticks = _align(live, ticks)
    aligned_close = _align(live, close)
    live_from_ticks = [b for _a, b in aligned_ticks]
    live_from_close = [b for _a, b in aligned_close]
    live_matched = [a for a, _b in aligned_ticks]

    report = {
        "preset": args.preset,
        "date": args.date,
        "window": f"{args.from_time}-{args.to_time}",
        "zEntry": z_entry,
        "live": _stats(live_matched, z_entry),
        "candleTicks": _stats(live_from_ticks, z_entry),
        "candleClose": _stats(live_from_close, z_entry),
        "liveFull": _stats([value for _s, value in live], z_entry),
        "candleTicksFull": _stats([value for _s, value in ticks], z_entry),
        "candleCloseFull": _stats([value for _s, value in close], z_entry),
    }
    report["ratios"] = {
        "stdTicksOverLive": round(
            report["candleTicks"]["std"] / report["live"]["std"], 3
        )
        if report["live"]["std"]
        else None,
        "p99TicksOverLive": round(report["candleTicks"]["p99"] / report["live"]["p99"], 3)
        if report["live"]["p99"]
        else None,
        "p99CloseOverLive": round(report["candleClose"]["p99"] / report["live"]["p99"], 3)
        if report["live"]["p99"]
        else None,
        # Главный показатель «раздутия» моста: хвосты синтетики против закрытий
        # на одном и том же дне и с одинаковой каденцией.
        "p99TicksOverClose": round(
            report["candleTicksFull"]["p99"] / report["candleCloseFull"]["p99"], 3
        )
        if report["candleCloseFull"]["p99"]
        else None,
        "p995TicksOverClose": round(
            report["candleTicksFull"]["p995"] / report["candleCloseFull"]["p995"], 3
        )
        if report["candleCloseFull"]["p995"]
        else None,
        "maxTicksOverClose": round(
            report["candleTicksFull"]["maxAbs"] / report["candleCloseFull"]["maxAbs"], 3
        )
        if report["candleCloseFull"]["maxAbs"]
        else None,
    }

    print(f"{args.preset} {args.date} {args.from_time}-{args.to_time}  (порог |Z| {z_entry})")
    header = f"{'источник':<16}{'точек':>8}{'std':>8}{'|Z|ср':>8}{'p95':>7}{'p99':>7}{'p99.5':>7}{'max':>7}{'>порог':>9}"
    print(header)
    print("-" * len(header))
    for name, key in (
        ("live (L2)", "live"),
        ("свечи ticks", "candleTicks"),
        ("свечи close", "candleClose"),
    ):
        row = report[key]
        print(
            f"{name:<16}{row['n']:>8}{row['std']:>8.3f}{row['absMean']:>8.3f}{row['p95']:>7.2f}"
            f"{row['p99']:>7.2f}{row['p995']:>7.2f}{row['maxAbs']:>7.2f}"
            f"{row['aboveEntry']:>5} ({row['aboveEntryPct']:.1f}%)"
        )
    print()
    print("полные ряды (без выравнивания):")
    for name, key in (
        ("live", "liveFull"),
        ("ticks", "candleTicksFull"),
        ("close", "candleCloseFull"),
    ):
        row = report[key]
        print(
            f"  {name:<8}{row['n']:>7} точек  std {row['std']:.3f}  p99 {row['p99']:.2f}  "
            f"max {row['maxAbs']:.2f}  выше порога {row['aboveEntry']} ({row['aboveEntryPct']:.2f}%)"
        )
    print()
    print(f"отношения: std(ticks)/std(live) = {report['ratios']['stdTicksOverLive']}, "
          f"p99(ticks)/p99(live) = {report['ratios']['p99TicksOverLive']}, "
          f"p99(close)/p99(live) = {report['ratios']['p99CloseOverLive']}")
    print(f"раздутие моста за день: p99(ticks)/p99(close) = {report['ratios']['p99TicksOverClose']}, "
          f"p99.5 = {report['ratios']['p995TicksOverClose']}, "
          f"max = {report['ratios']['maxTicksOverClose']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
