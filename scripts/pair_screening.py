"""Скрининг пар MOEX на 6-месячном наборе ISS (v2.5).

Проигрывает каждую пару через боевое ядро (`Premium`, `SingleLegLeadLag`,
`PrimeHoursOnly`, механизмы v2.4) и печатает таблицу: сделок в день, net, PF,
просадку и вердикт «включать / не включать».

Примеры:
    py -3 scripts/pair_screening.py                       # весь список по умолчанию
    py -3 scripts/pair_screening.py --pairs ROSN/LKOH TATN/LKOH
    py -3 scripts/pair_screening.py --from 2026-07-25 --to 2026-10-25   # 3 месяца
    py -3 scripts/pair_screening.py --json reports/pairs.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot_a.backtest.iss_source import fetch_pair  # noqa: E402
from bot_a.config import AppConfig, PairPreset  # noqa: E402
from bot_a.logging_setup import configure_logging  # noqa: E402
from bot_a.economics.fees import Tariff  # noqa: E402
from bot_a.models.domain import ExecutionTopology  # noqa: E402
from bot_a.services.paper_replay import PaperReplayService, session_window  # noqa: E402
from bot_a.utils.time_provider import MSK, msk_to_ms  # noqa: E402

DEFAULT_PAIRS: Tuple[str, ...] = (
    "VTBR/SBER",
    "SBERP/SBER",
    "GAZP/SBER",
    "ROSN/LKOH",
    "TATN/LKOH",
    "NVTK/GAZP",
    "LKOH/SBER",
    "ALRS/SBER",
)

SCREEN_MIN_PF = 1.25


def _preset(name: str, tickers: Sequence[str], futures: bool = False) -> PairPreset:
    leg1, leg2 = fetch_pair(tickers[0], tickers[1], futures=futures)
    return PairPreset(
        name=name,
        description=f"MOEX ISS research pair {tickers[0]}/{tickers[1]}",
        leg1=leg1,
        leg2=leg2,
        beta=1.0,
        beta_mode="slow_ratio",
        enabled=True,
        topology="SingleLegLeadLag",
        active_session_windows="PrimeHoursOnly",
    )


async def _run_pair(
    config: AppConfig,
    preset: PairPreset,
    start_ms: int,
    end_ms: int,
    warmup_days: int,
) -> Dict[str, Any]:
    service = PaperReplayService(config, preset)
    result, _warmup, _candles = await service.run(start_ms, end_ms, warmup_days=warmup_days)
    totals = result.totals
    days = max(totals.total_days, 1)
    return {
        "pair": preset.name,
        "leg1": preset.leg1.ticker,
        "leg2": preset.leg2.ticker,
        "lot1": preset.leg1.lot_size,
        "lot2": preset.leg2.lot_size,
        "price1": float(preset.leg1.rub_per_price_unit),
        "step1": str(preset.leg1.min_price_increment),
        "trades": int(totals.trades),
        "tradesPerDay": round(totals.trades / days, 2),
        "days": int(totals.total_days),
        "profitableDays": int(totals.profitable_days),
        "dayWinRatePct": round(100.0 * totals.profitable_days / days, 1),
        "grossRub": float(totals.gross_pnl_rub),
        "feesRub": float(totals.fees_rub),
        "netRub": float(totals.net_pnl_rub),
        "profitFactor": round(float(totals.profit_factor), 3),
        "maxDrawdownRub": float(totals.max_drawdown_rub),
        "avgHoldMin": round(totals.avg_hold_seconds / 60.0, 1),
        "counters": dict(result.counters),
    }


def _verdict(row: Dict[str, Any]) -> str:
    if row["trades"] == 0:
        return "нет сделок"
    if row["netRub"] > 0 and row["profitFactor"] > SCREEN_MIN_PF:
        return "ВКЛЮЧИТЬ"
    if row["netRub"] > 0:
        return f"мало PF (<{SCREEN_MIN_PF})"
    return "убыток"


def _print_table(rows: List[Dict[str, Any]]) -> None:
    header = (
        f"{'пара':<14}{'сдел.':>7}{'/день':>7}{'дней+':>8}{'win%':>7}"
        f"{'gross':>10}{'fees':>9}{'NET':>10}{'PF':>7}{'DD':>10}   вердикт"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda item: -item["netRub"]):
        print(
            f"{row['pair']:<14}{row['trades']:>7}{row['tradesPerDay']:>7.2f}"
            f"{row['profitableDays']:>4}/{row['days']:<3}{row['dayWinRatePct']:>7.1f}"
            f"{row['grossRub']:>10.3f}{row['feesRub']:>9.3f}{row['netRub']:>10.3f}"
            f"{row['profitFactor']:>7.3f}{row['maxDrawdownRub']:>10.3f}   {_verdict(row)}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Скрининг пар MOEX на ISS-данных")
    parser.add_argument("--pairs", nargs="*", default=list(DEFAULT_PAIRS), help="пары вида AAA/BBB")
    parser.add_argument("--from", dest="date_from", default="2026-07-25", help="начало (YYYY-MM-DD)")
    parser.add_argument("--to", dest="date_to", default="2026-10-25", help="конец (YYYY-MM-DD)")
    parser.add_argument("--warmup-days", type=int, default=5)
    parser.add_argument("--tariff", default="Premium", help="тариф T-Invest: Premium или Trader")
    parser.add_argument("--z-entry", type=float, default=2.8)
    parser.add_argument("--futures", action="store_true", help="пары на SPBFUT (FORTS)")
    parser.add_argument("--json", dest="json_out", default="")
    parser.add_argument("--config", default="appsettings.json")
    args = parser.parse_args(argv)

    configure_logging("WARNING", "logs", "pair_screening.log")
    config = AppConfig.load(args.config)
    config = config.with_overrides(tariff=Tariff(args.tariff))
    start_ms = msk_to_ms(datetime.fromisoformat(args.date_from).replace(tzinfo=MSK))
    end_ms = msk_to_ms(datetime.fromisoformat(args.date_to).replace(hour=23, minute=59, tzinfo=MSK))
    config = replace(
        config,
        risk=replace(config.risk, active_session_windows="PrimeHoursOnly"),
        strategy=config.strategy.with_overrides(
            z_entry=float(args.z_entry), use_entry_persistence=False
        ),
    )

    rows: List[Dict[str, Any]] = []
    for spec in args.pairs:
        tickers = [part.strip().upper() for part in spec.replace("/", " ").split()]
        if len(tickers) != 2:
            print(f"пропуск '{spec}': нужен формат AAA/BBB")
            continue
        name = f"{tickers[0]}_{tickers[1]}"
        try:
            preset = _preset(name, tickers, futures=bool(args.futures))
        except Exception as exc:  # noqa: BLE001 — ISS может не отдать тикер
            print(f"{name}: метаданные недоступны ({type(exc).__name__}: {exc})")
            continue
        try:
            row = asyncio.run(_run_pair(config, preset, start_ms, end_ms, args.warmup_days))
        except Exception as exc:  # noqa: BLE001 — нет свечей / сеть
            print(f"{name}: прогон не удался ({type(exc).__name__}: {exc})")
            continue
        rows.append(row)
        print(f"  {name}: net {row['netRub']:+.3f} RUB, PF {row['profitFactor']:.2f}, "
              f"сделок {row['trades']} ({row['tradesPerDay']}/день)")

    print()
    if rows:
        _print_table(rows)
        winners = [row["pair"] for row in rows if _verdict(row) == "ВКЛЮЧИТЬ"]
        print(f"\nКандидаты на включение (net > 0 и PF > {SCREEN_MIN_PF}): "
              f"{', '.join(winners) or '—'}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
