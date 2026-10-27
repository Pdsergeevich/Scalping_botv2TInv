"""Параллельный грид по парам/окнам/порогам на ISS-данных (v2.6).

Один прогон — это `HistoricalCandleBacktester` без записи отчётов на диск.
Конфигурации выполняются в пуле процессов, поэтому грид из десятков прогонов
укладывается в минуты.

Примеры:
    py -3 scripts/pair_grid.py --pairs VTBR/SBER ALRS/SBER \
        --ewma 900 1200 1600 2400 --z 2.6 2.8 --periods oos is 6m --jobs 8 \
        --json reports/ewma_grid_v26.json

    py -3 scripts/pair_grid.py --pairs AFLT/SBER AFKS/SBER --periods 6m --jobs 6
    py -3 scripts/pair_grid.py --pairs SPZ6/SRZ6 --futures --sigma 0.95 1.05 1.15 --periods 3m 6m
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PERIODS: Dict[str, Tuple[str, str]] = {
    "oos": ("2026-04-25", "2026-07-25"),
    "is": ("2026-07-25", "2026-10-25"),
    "6m": ("2026-04-25", "2026-10-25"),
    "3m": ("2026-07-25", "2026-10-25"),
}


def _resolve_presets(specs: Sequence[str], futures: bool) -> Dict[str, Any]:
    """Метаданные инструментов берём один раз в родителе и передаём в воркеры."""
    from bot_a.backtest.iss_source import fetch_pair
    from bot_a.config import PairPreset

    presets: Dict[str, Any] = {}
    for spec in specs:
        tickers = [part.strip().upper() for part in spec.replace("/", " ").split()]
        if len(tickers) != 2:
            print(f"пропуск '{spec}': нужен формат AAA/BBB")
            continue
        name = f"{tickers[0]}_{tickers[1]}"
        try:
            leg1, leg2 = fetch_pair(tickers[0], tickers[1], futures=futures)
        except Exception as exc:  # noqa: BLE001 — ISS может не знать тикер
            print(f"{name}: метаданные недоступны ({type(exc).__name__}: {exc})")
            continue
        presets[name] = PairPreset(
            name=name,
            description=f"grid {tickers[0]}/{tickers[1]}",
            leg1=leg1,
            leg2=leg2,
            beta=1.0,
            beta_mode="slow_ratio",
            enabled=True,
            topology="SingleLegLeadLag",
            active_session_windows="PrimeHoursOnly",
        )
    return presets


def _run_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Один прогон бэктеста. Живёт в отдельном процессе."""
    from bot_a.backtest.historical_candles import HistoricalCandleBacktester
    from bot_a.backtest.iss_source import IssCandleSource
    from bot_a.config import AppConfig
    from bot_a.economics.fees import Tariff
    from bot_a.utils.quotation import dec
    from bot_a.utils.time_provider import MSK, msk_to_ms

    logging.disable(logging.WARNING)
    config = AppConfig.load(str(ROOT / "appsettings.json"))
    config = config.with_overrides(tariff=Tariff(spec["tariff"]))
    config = replace(
        config,
        risk=replace(config.risk, active_session_windows="PrimeHoursOnly"),
        strategy=config.strategy.with_overrides(
            z_entry=float(spec["z"]),
            ewma_window=int(spec["ewma"]),
            use_entry_persistence=False,
            min_sigma_to_fee_ratio=float(spec["sigma"]),
            tp_deviation_fraction=float(spec.get("tp_dev", config.strategy.tp_deviation_fraction)),
            tp_min_fee_multiple=dec(
                spec.get("tp_fee", config.strategy.tp_min_fee_multiple)
            ),
            quantity_lots=max(int(spec.get("lots", config.strategy.quantity_lots)), 1),
        ),
    )
    start_ms = msk_to_ms(datetime.fromisoformat(spec["start"]).replace(tzinfo=MSK))
    end_ms = msk_to_ms(
        datetime.fromisoformat(spec["end"]).replace(hour=23, minute=59, tzinfo=MSK)
    )
    backtester = HistoricalCandleBacktester(
        config,
        spec["preset"],
        IssCandleSource(config.path(config.storage.data_cache_dir)),
    )
    try:
        result = asyncio.run(
            backtester.run(start_ms=start_ms, end_ms=end_ms, write_report=False)
        )
    except Exception as exc:  # noqa: BLE001 — нет свечей или сеть
        return {**{k: spec[k] for k in ("pair", "period", "ewma", "z", "sigma")},
                "error": f"{type(exc).__name__}: {exc}"}
    totals = result.totals
    days = max(totals.total_days, 1)
    return {
        "pair": spec["pair"],
        "period": spec["period"],
        "ewma": spec["ewma"],
        "z": spec["z"],
        "sigma": spec["sigma"],
        "tp_dev": spec.get("tp_dev", 0.85),
        "tp_fee": spec.get("tp_fee", 2.5),
        "lots": spec.get("lots", 1),
        "trades": int(totals.trades),
        "tradesPerDay": round(totals.trades / days, 2),
        "days": int(totals.total_days),
        "profitableDays": int(totals.profitable_days),
        "gross": float(totals.gross_pnl_rub),
        "fees": float(totals.fees_rub),
        "net": float(totals.net_pnl_rub),
        "pf": round(float(totals.profit_factor), 3),
        "dd": float(totals.max_drawdown_rub),
    }


def _print_table(rows: List[Dict[str, Any]]) -> None:
    ok = [row for row in rows if "error" not in row]
    bad = [row for row in rows if "error" in row]
    header = (
        f"{'пара':<12}{'период':>7}{'ewma':>7}{'z':>6}{'tpDev':>7}{'σ/Fee':>7}{'сдел.':>7}"
        f"{'/день':>7}{'gross':>10}{'fees':>9}{'NET':>10}{'PF':>7}{'DD':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(ok, key=lambda item: (item["pair"], item["period"], item["ewma"], item["z"])):
        print(
            f"{row['pair']:<12}{row['period']:>7}{row['ewma']:>7}{row['z']:>6.1f}"
            f"{row.get('tp_dev', 0.85):>7.2f}{row['sigma']:>7.2f}{row['trades']:>7}{row['tradesPerDay']:>7.2f}"
            f"{row['gross']:>10.3f}{row['fees']:>9.3f}{row['net']:>10.3f}"
            f"{row['pf']:>7.3f}{row['dd']:>9.3f}"
        )
    for row in bad:
        print(f"  {row['pair']} {row['period']} ewma={row['ewma']} z={row['z']}: {row['error']}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Параллельный грид бэктестов")
    parser.add_argument("--pairs", nargs="*", required=True, help="пары вида AAA/BBB")
    parser.add_argument("--ewma", nargs="*", type=int, default=None, help="окна EWMA")
    parser.add_argument("--z", nargs="*", type=float, default=None, help="пороги Z_entry")
    parser.add_argument("--sigma", nargs="*", type=float, default=None, help="minSigmaToFeeRatio")
    parser.add_argument("--tp-dev", nargs="*", type=float, default=None, help="tpDeviationFraction")
    parser.add_argument("--tp-fee", nargs="*", type=float, default=None, help="tpMinFeeMultiple")
    parser.add_argument("--lots", type=int, default=1, help="quantityLots (масштаб результата)")
    parser.add_argument(
        "--periods", nargs="*", default=["6m"], choices=sorted(PERIODS), help="периоды"
    )
    parser.add_argument("--tariff", default="Premium")
    parser.add_argument("--futures", action="store_true")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--json", dest="json_out", default="")
    args = parser.parse_args(argv)

    presets = _resolve_presets(args.pairs, futures=args.futures)
    if not presets:
        print("нет валидных пар")
        return 2
    ewma_values = args.ewma or [2400]
    z_values = args.z or [2.8]
    sigma_values = args.sigma or [1.30]
    tp_dev_values = args.tp_dev or [0.85]
    tp_fee_values = args.tp_fee or [2.5]

    specs: List[Dict[str, Any]] = []
    for period in args.periods:
        start, end = PERIODS[period]
        for name, preset in presets.items():
            for ewma in ewma_values:
                for z in z_values:
                    for sigma in sigma_values:
                        for tp_dev in tp_dev_values:
                            for tp_fee in tp_fee_values:
                                specs.append(
                                    {
                                        "pair": name,
                                        "preset": preset,
                                        "period": period,
                                        "start": start,
                                        "end": end,
                                        "ewma": ewma,
                                        "z": z,
                                        "sigma": sigma,
                                        "tp_dev": tp_dev,
                                        "tp_fee": tp_fee,
                                        "lots": args.lots,
                                        "tariff": args.tariff,
                                    }
                                )

    print(f"конфигураций: {len(specs)}, процессов: {args.jobs}")
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(args.jobs, 1)) as pool:
        futures = {pool.submit(_run_spec, spec): spec for spec in specs}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001
                spec = futures[future]
                rows.append(
                    {
                        "pair": spec["pair"],
                        "period": spec["period"],
                        "ewma": spec["ewma"],
                        "z": spec["z"],
                        "sigma": spec["sigma"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if done % 8 == 0:
                print(f"  готово {done}/{len(specs)}")

    print()
    _print_table(rows)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nJSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
