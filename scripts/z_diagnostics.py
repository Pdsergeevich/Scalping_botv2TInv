"""Разбор дня: насколько Z подходил к порогу входа и что именно мешало.

Скрипт проигрывает реальные 1-минутные свечи MOEX ISS через то же ядро, что
`--mode paper`, и печатает:

* максимум |Z| за день и время, когда он случился;
* почасовую таблицу (сколько подтиков было, максимум |Z|, сколько раз |Z| >= Z_entry);
* список моментов, когда сигнал подошёл ближе всего, с чек-листом 8 фильтров;
* счётчики пропусков из ФСМ (hook / regime / obi / session / vol / beta-drift ...).

Примеры:
    py -3 scripts/z_diagnostics.py --date 2026-10-26
    py -3 scripts/z_diagnostics.py --date 2026-10-26 --preset SBERP_SBER --tariff Trader
    py -3 scripts/z_diagnostics.py --from 2026-10-20 --to 2026-10-26 --top 8
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot_a.backtest.historical_candles import iter_candle_events  # noqa: E402
from bot_a.backtest.iss_source import IssCandleSource  # noqa: E402
from bot_a.cli import _apply_cli_overrides, build_parser, _parse_period  # noqa: E402
from bot_a.config import AppConfig  # noqa: E402
from bot_a.economics.fees import TinkoffFeeCalculator  # noqa: E402
from bot_a.engine.factory import create_engine, create_guard  # noqa: E402
from bot_a.engine.fsm import StrategyParams  # noqa: E402
from bot_a.engine.ledger import PortfolioLedger  # noqa: E402
from bot_a.execution.simulated_gateway import SimulatedOrderExecutionGateway  # noqa: E402
from bot_a.services.bootstrap import bootstrap_warmup  # noqa: E402
from bot_a.services.live_dashboard import LiveWatchService  # noqa: E402
from bot_a.logging_setup import configure_logging  # noqa: E402
from bot_a.utils.time_provider import MSK, SimulatedTimeProvider  # noqa: E402

FILTER_TITLES = {
    "z": "|Z| >= порога входа",
    "hook": "хук пика Z (подтверждение разворота)",
    "regime": "режимный фильтр (Leg2 ведёт)",
    "obi": "поток стакана OBI",
    "vol": "sigma / комиссия",
    "edge": "покрытие комиссии возвратом",
    "window": "окно входа (activeSessionWindows)",
    "daily": "дневной предохранитель",
}


def _msk(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=MSK).strftime("%d.%m %H:%M:%S")


class _HourCell:
    __slots__ = ("count", "zmin", "zmax", "over", "first_ms")

    def __init__(self, first_ms: int) -> None:
        self.count = 0
        self.zmin = float("inf")
        self.zmax = float("-inf")
        self.over = 0
        self.first_ms = first_ms


class _DayStats:
    """Копит статистику Z по часам и хранит лучшие моменты для разбора."""

    def __init__(self, z_entry: float, top: int) -> None:
        self._z_entry = z_entry
        self._top = top
        self._hours: Dict[int, _HourCell] = {}
        self._best: List[Tuple[float, int, Dict[str, Any]]] = []
        self._zmin = float("inf")
        self._zmax = float("-inf")
        self._samples = 0
        self._candidates = 0
        self._filter_blocks: Counter = Counter()
        self._filter_totals: Counter = Counter()

    def feed(self, metrics: Any, diagnostics: Optional[Dict[str, Any]] = None) -> None:
        if metrics is None:
            return
        z = float(metrics.zscore)
        az = abs(z)
        ts = int(metrics.time_ms)
        self._samples += 1
        self._zmin = min(self._zmin, z)
        self._zmax = max(self._zmax, z)

        cell = self._hours.get(ts // 3_600_000)
        if cell is None:
            cell = self._hours[ts // 3_600_000] = _HourCell(ts)
        cell.count += 1
        cell.zmin = min(cell.zmin, z)
        cell.zmax = max(cell.zmax, z)
        if az >= self._z_entry:
            cell.over += 1
            self._candidates += 1
            if diagnostics is not None:
                for name, ok in diagnostics.get("checks", {}).items():
                    self._filter_totals[name] += 1
                    if not ok:
                        self._filter_blocks[name] += 1
        if diagnostics is not None and az >= self._z_entry * 0.6:
            self._best.append((az, ts, diagnostics))
            self._best.sort(key=lambda item: -item[0])
            del self._best[self._top :]

    def report(self, counters: Dict[str, int], z_entry: float, trades: int) -> str:
        out: List[str] = []
        out.append("=" * 96)
        out.append("ПОЧАСОВАЯ РАСКЛАДКА Z (реальные свечи MOEX ISS через боевое ядро)")
        out.append("=" * 96)
        out.append(f"{'час MSK':<18}{'замеров':>9}{'Z min':>9}{'Z max':>9}{'|Z|>=порог':>13}   диапазон")
        for hour_ms in sorted(self._hours):
            cell = self._hours[hour_ms]
            label = datetime.fromtimestamp(cell.first_ms / 1000, tz=MSK).strftime("%d.%m %H:00")
            bar_len = int(round(max(abs(cell.zmax), abs(cell.zmin)) / max(z_entry, 0.01) * 20))
            bar = "#" * bar_len
            out.append(
                f"{label:<18}{cell.count:>9}{cell.zmin:>9.2f}{cell.zmax:>9.2f}{cell.over:>13}   {bar}"
            )
        out.append("=" * 96)
        if self._samples:
            share = 100.0 * self._candidates / self._samples
            out.append(
                f"ИТОГ: замеров {self._samples}, Z в диапазоне [{self._zmin:+.2f}; {self._zmax:+.2f}], "
                f"макс |Z| = {max(abs(self._zmin), abs(self._zmax)):.2f} при пороге {z_entry:.2f}, "
                f"выше порога {self._candidates} замеров ({share:.2f}% времени), сделок {trades}"
            )
        out.append("=" * 96)

        active = {k: v for k, v in counters.items() if v}
        if active:
            out.append("Счётчики ФСМ (сколько раз что срабатывало):")
            for name, value in sorted(active.items(), key=lambda kv: -kv[1]):
                out.append(f"  {name:<24}{value}")
        else:
            out.append(
                "Счётчики ФСМ пусты: ядро ни разу не дошло до стадии сигнала — "
                "|Z| не превысил порог входа."
            )
        out.append("")

        if self._filter_totals:
            out.append("Если |Z| уже был выше порога, что мешало (доля блокировок):")
            for name, total in self._filter_totals.most_common():
                blocked = self._filter_blocks[name]
                share = 100.0 * blocked / total if total else 0.0
                out.append(f"  {FILTER_TITLES.get(name, name):<44}{blocked:>6} из {total:<6} ({share:5.1f}%)")
            out.append("")

        out.append(f"БЛИЖАЙШИЕ К ВХОДУ МОМЕНТЫ (топ-{len(self._best)} по |Z|):")
        if not self._best:
            out.append("  нет замеров выше 60% порога — день был полностью вне зоны входа")
        for az, ts, diag in self._best:
            missing = diag.get("missing") or []
            state = "ВХОД РАЗРЕШЁН" if diag.get("ready") else "заблокирован: " + "; ".join(missing)
            out.append(f"  {_msk(ts)}  |Z|={az:.2f}  {state}")
            for name, ok in diag.get("checks", {}).items():
                mark = "OK  " if ok else "НЕТ "
                out.append(
                    f"      [{mark}] {FILTER_TITLES.get(name, name):<42}"
                    f"{diag.get('details', {}).get(name, '')}"
                )
        return "\n".join(out)


async def _run(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    config = _apply_cli_overrides(config, args)
    logger = configure_logging(config.logging.level, str(config.path(config.logging.log_dir)), config.logging.file_name)
    preset = config.preset()
    start_ms, end_ms = _parse_period(args)
    if start_ms is None or end_ms is None:
        print("нужны --date или --from/--to")
        return 2

    params: StrategyParams = config.strategy
    clock = SimulatedTimeProvider(start_ms)
    fee_calculator = TinkoffFeeCalculator(config.fees.tariff, clock=clock)
    gateway = SimulatedOrderExecutionGateway(
        metadata=[preset.leg1, preset.leg2],
        clock=clock,
        fee_calculator=fee_calculator,
        latency_ms=config.execution.simulated_order_latency_ms,
        queue_fill_factor=config.execution.queue_fill_factor,
        slippage_steps=config.execution.slippage_steps,
        order_book_depth=config.execution.order_book_depth,
        initial_cash_rub=config.execution.paper_initial_cash_rub,
    )
    guard = create_guard(config, [preset.leg1, preset.leg2])
    ledger = PortfolioLedger(config.execution.paper_initial_cash_rub)
    engine = create_engine(
        [preset.leg1, preset.leg2],
        params,
        config.execution.topology,
        gateway,
        fee_calculator,
        clock,
        guard,
        ledger,
        logger,
    )
    await gateway.start()
    source = IssCandleSource(config.path(config.storage.data_cache_dir))
    if args.warmup_days > 0:
        await bootstrap_warmup(
            config,
            preset,
            engine,
            source=source,
            days=args.warmup_days,
            end_ms=start_ms - 1,
            logger=logger,
            clock=clock,
        )
    candles1, candles2 = await asyncio.gather(
        source.get_candles(preset.leg1, start_ms, end_ms, "1min"),
        source.get_candles(preset.leg2, start_ms, end_ms, "1min"),
    )
    if not candles1 or not candles2:
        print("нет свечей за период — проверьте даты и кэш")
        return 2

    stats = _DayStats(params.z_entry, args.top)
    diagnostics = LiveWatchService._entry_diagnostics  # noqa: SLF001 - только чтение состояния ядра
    events = iter_candle_events(
        preset.leg1,
        preset.leg2,
        list(candles1),
        list(candles2),
        depth=config.execution.order_book_depth,
        phase_lock=True,
        synth_mode="ticks",
    )
    last_sample_ms = -1
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
        # Z считаем на каждом подтике, чек-лист — раз в 5 секунд (как в боевом режиме).
        if int(metrics.time_ms) - last_sample_ms >= 5_000:
            last_sample_ms = int(metrics.time_ms)
            diag = diagnostics(None, engine, metrics)
            stats.feed(metrics, diag)
        else:
            stats.feed(metrics, None)

    await engine.shutdown(flatten=True)
    await gateway.stop()
    print(stats.report(dict(engine.counters), params.z_entry, len(ledger.trades)))
    if ledger.trades:
        print("\nСДЕЛКИ:")
        for trade in ledger.trades:
            opened = datetime.fromtimestamp(trade.opened_ms / 1000, tz=MSK).strftime("%H:%M:%S")
            closed = datetime.fromtimestamp(trade.closed_ms / 1000, tz=MSK).strftime("%H:%M:%S")
            print(
                f"  {opened} -> {closed}  {trade.side_leg1.value:<5} "
                f"вход {trade.leg1_entry_price} -> выход {trade.leg1_exit_price}  "
                f"Z {trade.entry_z:+.2f} -> {trade.exit_z:+.2f}  "
                f"причина {trade.exit_reason.value:<14} "
                f"gross {trade.gross_pnl_rub:+.4f} fees {trade.fees_rub:.4f} "
                f"net {trade.net_pnl_rub:+.4f} RUB  удержание {trade.hold_seconds / 60:.0f} мин"
            )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    parser.description = "Диагностика Z-скора: почему не было входов"
    parser.add_argument("--date", help="один торговый день (YYYY-MM-DD)")
    parser.add_argument("--top", type=int, default=6, help="сколько ближайших моментов показать")
    parser.add_argument("--warmup-days", type=int, default=5, help="дней прогрева перед сессией")
    args = parser.parse_args(["--mode", "paper"] + raw)
    args.source = "iss"
    if args.date:
        args.date_from = args.date_from or args.date
        args.date_to = args.date_to or args.date
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
