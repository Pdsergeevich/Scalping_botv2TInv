"""Параллельный grid-search по параметрам стратегии."""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bot_a.backtest.candles import CandleSource
from bot_a.backtest.historical_candles import iter_candle_events
from bot_a.backtest.reporting import write_json
from bot_a.backtest.simulation import simulate_events
from bot_a.backtest.synthetics import TICKS_PER_MINUTE
from bot_a.config import AppConfig, PairPreset
from bot_a.engine.fsm import StrategyParams
from bot_a.models.domain import BacktestTotals, DailyBacktestSummary, ExecutionTopology
from bot_a.utils.quotation import fmt_rub
from bot_a.utils.time_provider import now_ms

LOGGER_NAME = "bot_a.optimize"

DEFAULT_Z_ENTRIES: Tuple[float, ...] = (1.8, 2.0, 2.2, 2.5, 2.8)
DEFAULT_Z_EXITS: Tuple[float, ...] = (0.0, 0.2, 0.4)
DEFAULT_EWMA_WINDOWS: Tuple[int, ...] = (600, 1200, 2400)
DEFAULT_TOPOLOGIES: Tuple[ExecutionTopology, ...] = (
    ExecutionTopology.DUAL_LEG_HEDGED,
    ExecutionTopology.SINGLE_LEG_LEAD_LAG,
)


def _null_logger(name: str = "bot_a.optimize.silent") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    return logger


@dataclass(frozen=True, slots=True)
class GridSearchResult:
    z_entry: float
    z_exit: float
    ewma_window: int
    topology: ExecutionTopology
    params: StrategyParams
    totals: BacktestTotals
    summaries: Tuple[DailyBacktestSummary, ...]

    @property
    def profitable_day_ratio(self) -> float:
        if self.totals.total_days == 0:
            return 0.0
        return self.totals.profitable_days / self.totals.total_days

    @property
    def score(self) -> Tuple[float, float, float]:
        """Ключ ранжирования: чистый PnL, доля прибыльных дней, Sharpe."""
        return (float(self.totals.net_pnl_rub), self.profitable_day_ratio, self.totals.sharpe_annual)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zEntry": self.z_entry,
            "zExit": self.z_exit,
            "ewmaWindow": self.ewma_window,
            "topology": self.topology.value,
            "netPnlRub": str(self.totals.net_pnl_rub),
            "grossPnlRub": str(self.totals.gross_pnl_rub),
            "feesRub": str(self.totals.fees_rub),
            "trades": self.totals.trades,
            "profitableDays": f"{self.totals.profitable_days}/{self.totals.total_days}",
            "winRatePct": round(self.totals.win_rate_pct, 2),
            "profitFactor": round(self.totals.profit_factor, 3),
            "sharpeAnnual": round(self.totals.sharpe_annual, 3),
            "maxDrawdownRub": str(self.totals.max_drawdown_rub),
            "feeDragPct": round(self.totals.fee_drag_pct, 2),
            "params": self.params.to_dict(),
        }


def _worker(payload: Tuple[Any, ...]) -> Dict[str, Any]:
    """Прогон одной комбинации в отдельном процессе (Windows spawn-совместимо)."""
    config, preset, candles1, candles2, params, topology, ticks, depth, start_ms = payload
    events = iter_candle_events(preset.leg1, preset.leg2, candles1, candles2, ticks, depth)
    outcome = asyncio.run(
        simulate_events(config, preset, events, params, topology, start_ms, _null_logger())
    )
    totals = outcome.ledger.totals()
    return {
        "z_entry": params.z_entry,
        "z_exit": params.z_exit,
        "ewma_window": params.ewma_window,
        "topology": topology,
        "params": params,
        "totals": totals,
        "summaries": tuple(outcome.ledger.daily_summaries()),
        "events": outcome.events_processed,
    }


class GridSearchOptimizer:
    """Перебор ``Z_Entry x Z_Exit x EwmaWindow x ExecutionTopology``."""

    def __init__(
        self,
        config: AppConfig,
        preset: PairPreset,
        source: CandleSource,
        base_params: Optional[StrategyParams] = None,
        jobs: Optional[int] = None,
        ticks_per_minute: int = TICKS_PER_MINUTE,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._preset = preset
        self._source = source
        self._base_params = base_params or config.strategy
        self._jobs = jobs if jobs is not None else max((os.cpu_count() or 1) - 1, 1)
        self._ticks = ticks_per_minute
        self._log = logger or logging.getLogger(LOGGER_NAME)

    async def run(
        self,
        days: int = 7,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        interval: str = "1min",
        z_entries: Sequence[float] = DEFAULT_Z_ENTRIES,
        z_exits: Sequence[float] = DEFAULT_Z_EXITS,
        ewma_windows: Sequence[int] = DEFAULT_EWMA_WINDOWS,
        topologies: Sequence[ExecutionTopology] = DEFAULT_TOPOLOGIES,
        write_reports: bool = True,
    ) -> List[GridSearchResult]:
        finish = end_ms if end_ms is not None else now_ms()
        start = start_ms if start_ms is not None else finish - days * 86_400_000
        candles1, candles2 = await asyncio.gather(
            self._source.get_candles(self._preset.leg1, start, finish, interval),
            self._source.get_candles(self._preset.leg2, start, finish, interval),
        )
        if not candles1 or not candles2:
            raise RuntimeError("Нет свечей для оптимизации: проверьте период и токен")
        combinations: List[Tuple[StrategyParams, ExecutionTopology]] = []
        for z_entry in z_entries:
            for z_exit in z_exits:
                for window in ewma_windows:
                    params = replace(
                        self._base_params,
                        z_entry=float(z_entry),
                        z_exit=float(z_exit),
                        ewma_window=int(window),
                    )
                    for topology in topologies:
                        combinations.append((params, topology))
        self._log.info(
            "Оптимизация %s: %d комбинаций, %d процесс(ов), свечей %d/%d",
            self._preset.name,
            len(combinations),
            self._jobs,
            len(candles1),
            len(candles2),
        )
        depth = self._config.execution.order_book_depth
        results: List[GridSearchResult] = []
        if self._jobs <= 1 or len(combinations) == 1:
            for index, (params, topology) in enumerate(combinations, start=1):
                outcome = await simulate_events(
                    self._config,
                    self._preset,
                    iter_candle_events(
                        self._preset.leg1, self._preset.leg2, candles1, candles2, self._ticks, depth
                    ),
                    params,
                    topology,
                    start,
                    _null_logger(),
                )
                results.append(
                    GridSearchResult(
                        z_entry=params.z_entry,
                        z_exit=params.z_exit,
                        ewma_window=params.ewma_window,
                        topology=topology,
                        params=params,
                        totals=outcome.ledger.totals(),
                        summaries=tuple(outcome.ledger.daily_summaries()),
                    )
                )
                self._log.info("Комбинация %d/%d готова", index, len(combinations))
        else:
            payloads = [
                (self._config, self._preset, list(candles1), list(candles2), params, topology, self._ticks, depth, start)
                for params, topology in combinations
            ]
            loop = asyncio.get_running_loop()
            with ProcessPoolExecutor(max_workers=min(self._jobs, len(payloads))) as pool:
                raw_results = await loop.run_in_executor(None, lambda: list(pool.map(_worker, payloads)))
            for item in raw_results:
                results.append(
                    GridSearchResult(
                        z_entry=float(item["z_entry"]),
                        z_exit=float(item["z_exit"]),
                        ewma_window=int(item["ewma_window"]),
                        topology=item["topology"],
                        params=item["params"],
                        totals=item["totals"],
                        summaries=tuple(item["summaries"]),
                    )
                )
        results.sort(key=lambda item: item.score, reverse=True)
        self._log.info("\n%s", self.render_table(results))
        if write_reports and results:
            self._write_reports(results, start, finish)
        return results

    def render_table(self, results: Sequence[GridSearchResult], limit: int = 12) -> str:
        header = (
            f"{'Z_Entry':>8}{'Z_Exit':>8}{'EWMA':>7}{'Топология':>18}"
            f"{'Сделок':>8}{'Дней':>9}{'Net RUB':>14}{'Sharpe':>9}{'MaxDD':>12}"
        )
        lines = [header, "-" * len(header)]
        for result in results[:limit]:
            days = f"{result.totals.profitable_days}/{result.totals.total_days}"
            lines.append(
                f"{result.z_entry:>8.2f}"
                f"{result.z_exit:>8.2f}"
                f"{result.ewma_window:>7d}"
                f"{result.topology.value:>18}"
                f"{result.totals.trades:>8d}"
                f"{days:>9}"
                f"{fmt_rub(result.totals.net_pnl_rub):>14}"
                f"{result.totals.sharpe_annual:>9.2f}"
                f"{fmt_rub(result.totals.max_drawdown_rub):>12}"
            )
        if not results:
            lines.append("(нет результатов)")
        return "\n".join(lines)

    def _write_reports(self, results: Sequence[GridSearchResult], start_ms: int, finish_ms: int) -> None:
        reports_dir = self._config.path(self._config.storage.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        winner = results[0]
        winner_path = reports_dir / f"optimize_winner_{self._preset.name}_{stamp}.json"
        write_json(
            winner_path,
            {
                "preset": self._preset.name,
                "generatedAt": datetime.now().isoformat(),
                "period": {"startMs": start_ms, "endMs": finish_ms},
                "winner": winner.to_dict(),
                "top": [item.to_dict() for item in results[:10]],
            },
        )
        table_path = reports_dir / f"optimize_{self._preset.name}_{stamp}.md"
        lines = [
            f"# Оптимизация {self._preset.name}",
            "",
            f"Период: {start_ms} .. {finish_ms} (UTC ms)",
            "",
            "| Z_Entry | Z_Exit | EWMA | Топология | Сделок | Прибыльных дней | Net, RUB | Sharpe | Max DD, RUB |",
            "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for result in results:
            lines.append(
                f"| {result.z_entry:.2f} | {result.z_exit:.2f} | {result.ewma_window} | {result.topology.value} |"
                f" {result.totals.trades} | {result.totals.profitable_days}/{result.totals.total_days} |"
                f" {fmt_rub(result.totals.net_pnl_rub)} | {result.totals.sharpe_annual:.2f} |"
                f" {fmt_rub(result.totals.max_drawdown_rub)} |"
            )
        table_path.write_text("\n".join(lines), encoding="utf-8")
        self._log.info("Победившая конфигурация: %s", winner_path)
        self._log.info("Полная таблица: %s", table_path)


