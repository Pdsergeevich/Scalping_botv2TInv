"""Симулятор портфеля: общая эквити, PF, Sharpe и просадка (v2.6.2).

Прогоняет несколько пар через боевое ядро, масштабирует результат на заданное число
лотов (выравнивание номинала) и складывает дневные PnL в общую кривую.

Примеры:
    py -3 scripts/portfolio_sim.py                       # портфель из appsettings
    py -3 scripts/portfolio_sim.py --notional 3000       # по 3 000 RUB на пару
    py -3 scripts/portfolio_sim.py --specs VTBR_SBER:2400:2.8:1 MTLR_SBER:900:2.8:1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot_a.utils.process_guard import (  # noqa: E402 — нужен sys.path выше
    hard_limit,
    limit_blas_threads,
    lower_priority,
    resolve_jobs,
    worker_init,
)

limit_blas_threads(1)

PERIODS: Dict[str, Tuple[str, str]] = {
    "oos": ("2026-04-25", "2026-07-25"),
    "is": ("2026-07-25", "2026-10-25"),
    "6m": ("2026-04-25", "2026-10-25"),
}


def _run_leg(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Один прогон пары; возвращает сделки и дневные итоги (для композиции портфеля)."""
    from bot_a.backtest.historical_candles import HistoricalCandleBacktester
    from bot_a.backtest.iss_source import IssCandleSource
    from bot_a.config import AppConfig
    from bot_a.economics.fees import Tariff
    from bot_a.models.domain import ExecutionTopology
    from bot_a.utils.quotation import dec
    from bot_a.utils.time_provider import MSK, msk_to_ms

    logging.disable(logging.WARNING)
    config = AppConfig.load(str(ROOT / "appsettings.json"))
    config = config.with_overrides(tariff=Tariff(spec["tariff"]))
    config = replace(
        config,
        risk=replace(config.risk, active_session_windows="PrimeHoursOnly"),
        execution=replace(config.execution, topology=ExecutionTopology(spec["topology"])),
        strategy=config.strategy.with_overrides(
            z_entry=float(spec["z"]),
            ewma_window=int(spec["ewma"]),
            use_entry_persistence=False,
            min_sigma_to_fee_ratio=float(spec["sigma"]),
            tp_deviation_fraction=float(spec["tp_dev"]),
            tp_min_fee_multiple=dec(spec["tp_fee"]),
            quantity_lots=1,  # масштабируем результат вручную: лоты не меняют форму сделки
        ),
    )
    preset = spec["preset"]
    start_ms = msk_to_ms(datetime.fromisoformat(spec["start"]).replace(tzinfo=MSK))
    end_ms = msk_to_ms(
        datetime.fromisoformat(spec["end"]).replace(hour=23, minute=59, tzinfo=MSK)
    )
    backtester = HistoricalCandleBacktester(
        config, preset, IssCandleSource(config.path(config.storage.data_cache_dir))
    )
    try:
        result = asyncio.run(backtester.run(start_ms=start_ms, end_ms=end_ms, write_report=False))
    except Exception as exc:  # noqa: BLE001
        return {"pair": spec["pair"], "error": f"{type(exc).__name__}: {exc}"}

    days = [
        {"date": str(getattr(summary, "date", "")), "net": float(summary.net_pnl_rub)}
        for summary in result.summaries
    ]
    trades = [
        {
            "net": float(trade.net_pnl_rub),
            "gross": float(trade.gross_pnl_rub),
            "fees": float(trade.fees_rub),
        }
        for trade in result.trades
    ]
    return {
        "pair": spec["pair"],
        "lots": int(spec["lots"]),
        "days": days,
        "trades": trades,
        "trades_total": len(trades),
        "net_total": sum(item["net"] for item in trades),
        "days_total": len(days),
    }


def _load_presets(names: Sequence[str]) -> Dict[str, Any]:
    from bot_a.backtest.iss_source import fetch_pair
    from bot_a.config import AppConfig, PairPreset

    config = AppConfig.load(str(ROOT / "appsettings.json"))
    presets: Dict[str, Any] = {}
    for name in names:
        try:
            preset = config.preset(name)
        except KeyError:
            tickers = name.split("_")
            if len(tickers) != 2:
                print(f"пропуск {name}: неизвестный пресет")
                continue
            leg1, leg2 = fetch_pair(tickers[0], tickers[1], futures=False)
            preset = PairPreset(
                name=name,
                description="",
                leg1=leg1,
                leg2=leg2,
                beta=1.0,
                beta_mode="slow_ratio",
                topology="SingleLegLeadLag",
                active_session_windows="PrimeHoursOnly",
            )
        presets[name] = preset
    return presets


def _notional_per_lot(preset: Any) -> float:
    """Рублёвая стоимость 1 лота по последней известной цене из кэша свечей."""
    from bot_a.backtest.iss_source import IssCandleSource
    from bot_a.config import AppConfig
    from bot_a.utils.time_provider import MSK, msk_to_ms

    config = AppConfig.load(str(ROOT / "appsettings.json"))
    source = IssCandleSource(config.path(config.storage.data_cache_dir))
    cache_dir = Path(config.path(config.storage.data_cache_dir))
    files = sorted(cache_dir.glob(f"iss_*_{preset.leg1.ticker}_*.json"))
    price = 0.0
    if files:
        try:
            rows = json.loads(files[-1].read_text(encoding="utf-8"))
            if rows:
                price = float(rows[-1]["close"])
        except (OSError, ValueError, KeyError):
            price = 0.0
    return price * float(preset.leg1.rub_per_price_unit)


def _stats(daily: Dict[str, float]) -> Dict[str, float]:
    values = [daily[key] for key in sorted(daily)]
    if not values:
        return {"days": 0, "net": 0.0, "sharpe": 0.0, "dd": 0.0, "greenPct": 0.0}
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    sharpe = (mean / std * math.sqrt(252)) if std > 0 else 0.0
    green = sum(1 for value in values if value > 0)
    return {
        "days": len(values),
        "net": round(equity, 4),
        "sharpe": round(sharpe, 2),
        "dd": round(drawdown, 4),
        "greenPct": round(100.0 * green / len(values), 1),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Портфельная симуляция с выравниванием номинала")
    parser.add_argument(
        "--specs",
        nargs="*",
        default=None,
        help="явные спеки: ПАРА:ewma:z[:tp_dev[:tp_fee[:lots]]] (иначе берём активные пресеты)",
    )
    parser.add_argument("--pairs", nargs="*", default=None, help="только эти активные пресеты")
    parser.add_argument("--period", default="6m", choices=sorted(PERIODS))
    parser.add_argument("--tariff", default="Premium")
    parser.add_argument("--notional", type=float, default=0.0, help="целевой номинал на пару, RUB")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=f"процессов (по умолчанию 1, максимум {hard_limit()})",
    )
    parser.add_argument("--json", dest="json_out", default="")
    args = parser.parse_args(argv)

    if args.specs:
        names = sorted({item.split(":")[0] for item in args.specs})
    else:
        from bot_a.config import AppConfig

        config = AppConfig.load(str(ROOT / "appsettings.json"))
        names = list(args.pairs or config.trading_preset_names())
        args.specs = []
        for name in names:
            preset = config.preset(name)
            args.specs.append(
                f"{name}:{preset.ewma_window or config.strategy.ewma_window}:"
                f"{preset.z_entry or config.strategy.z_entry}"
                + (f":{preset.tp_deviation_fraction}" if preset.tp_deviation_fraction else "")
            )

    presets = _load_presets(names)
    start, end = PERIODS[args.period]
    specs: List[Dict[str, Any]] = []
    for raw in args.specs:
        parts = raw.split(":")
        name = parts[0]
        if name not in presets:
            continue
        ewma = int(parts[1]) if len(parts) > 1 and parts[1] else 2400
        z = float(parts[2]) if len(parts) > 2 and parts[2] else 2.8
        tp_dev = float(parts[3]) if len(parts) > 3 and parts[3] else 0.85
        tp_fee = float(parts[4]) if len(parts) > 4 and parts[4] else 2.5
        lots = int(parts[5]) if len(parts) > 5 and parts[5] else 1
        preset = presets[name]
        notional = _notional_per_lot(preset)
        if args.notional > 0 and notional > 0:
            lots = max(int(round(args.notional / notional)), 1)
        specs.append(
            {
                "pair": name,
                "preset": preset,
                "start": start,
                "end": end,
                "ewma": ewma,
                "z": z,
                "sigma": float(preset.min_sigma_to_fee_ratio or 1.30),
                "tp_dev": tp_dev,
                "tp_fee": tp_fee,
                "lots": lots,
                "topology": preset.topology or "SingleLegLeadLag",
                "tariff": args.tariff,
                "notional_per_lot": round(notional, 2),
            }
        )

    if not specs:
        print("нет валидных спек")
        return 2
    print(f"пар в портфеле: {len(specs)}")
    jobs = resolve_jobs(args.jobs)
    print(f"процессов: {jobs}")
    lower_priority()
    runs: List[Dict[str, Any]] = []
    if jobs <= 1:
        for spec in specs:
            try:
                row = _run_leg(spec)
            except Exception as exc:  # noqa: BLE001
                row = {"pair": spec["pair"], "error": f"{type(exc).__name__}: {exc}"}
            row["notional_per_lot"] = spec.get("notional_per_lot", 0.0)
            runs.append(row)
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=worker_init) as pool:
            futures = {pool.submit(_run_leg, spec): spec for spec in specs}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = {"pair": spec["pair"], "error": f"{type(exc).__name__}: {exc}"}
                row["notional_per_lot"] = spec.get("notional_per_lot", 0.0)
                runs.append(row)

    combined: Dict[str, float] = defaultdict(float)
    total_trades = 0
    gross_win = 0.0
    gross_loss = 0.0
    print(f"\n{'пара':<14}{'лотов':>7}{'номинал':>10}{'сдел.':>7}{'NET RUB':>11}{'DD RUB':>9}{'Sharpe':>8}{'зелёных':>9}")
    for row in sorted(runs, key=lambda item: item["pair"]):
        if "error" in row:
            print(f"{row['pair']:<14}  ошибка: {row['error']}")
            continue
        lots = max(int(row["lots"]), 1)
        net = row["net_total"] * lots
        for day in row["days"]:
            combined[day["date"]] += day["net"] * lots
        for trade in row["trades"]:
            value = trade["net"] * lots
            if value >= 0:
                gross_win += value
            else:
                gross_loss += -value
        total_trades += row["trades_total"]
        daily = {day["date"]: day["net"] * lots for day in row["days"]}
        stats = _stats(daily)
        print(
            f"{row['pair']:<14}{lots:>7}{row.get('notional_per_lot', 0) * lots:>10.0f}"
            f"{row['trades_total']:>7}{net:>11.3f}{stats['dd']:>9.3f}"
            f"{stats['sharpe']:>8.2f}{stats['greenPct']:>8.1f}%"
        )

    portfolio = _stats(dict(combined))
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    days = portfolio["days"] or 1
    print("\n" + "=" * 84)
    print(f"ПОРТФЕЛЬ: сделок {total_trades} ({total_trades / days:.2f}/день), дней {portfolio['days']}")
    print(f"  Net         {portfolio['net']:+.4f} RUB")
    print(f"  PF          {pf:.3f}")
    print(f"  Sharpe      {portfolio['sharpe']:.2f}")
    print(f"  Max DD      {portfolio['dd']:.4f} RUB")
    print(f"  Зелёных дней {portfolio['greenPct']:.1f}%")
    print("=" * 84)

    if args.json_out:
        payload = {
            "period": args.period,
            "runs": [{k: v for k, v in row.items() if k not in ("days", "trades")} for row in runs],
            "portfolio": {**portfolio, "pf": pf, "trades": total_trades},
            "daily": dict(sorted(combined.items())),
        }
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
