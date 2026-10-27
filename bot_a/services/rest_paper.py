"""Paper-торговля на живом L2-потоке T-Invest через REST (без gRPC SDK).

Python 3.14 не имеет сборок ``grpcio``, поэтому ``TinkoffMarketDataStream`` недоступен.
Этот сервис опрашивает ``MarketDataService/GetOrderBook`` (глубина 20) с заданным
интервалом 100–250 мс и ``GetLastTrades`` для принтов, кормит **то же** ядро и
исполняет заявки через ``SimulatedOrderExecutionGateway`` — реальные деньги не тратятся.

Статистические окна двигаются раз в ``statSampleIntervalSeconds`` (v2.4.1), а
котирование/стопы реагируют на каждый пакет стакана.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from bot_a.broker.tinkoff_rest_client import TInvestRestClient, TInvestRestError
from bot_a.config import AppConfig, PairPreset
from bot_a.economics.fees import TinkoffFeeCalculator
from bot_a.engine.factory import create_engine, create_guard
from bot_a.engine.fsm import StrategyParams
from bot_a.engine.ledger import PortfolioLedger
from bot_a.execution.simulated_gateway import SimulatedOrderExecutionGateway
from bot_a.models.domain import ExecutionTopology, MarketEvent
from bot_a.services.l2_sync import PairBookSyncer
from bot_a.utils.time_provider import MSK, RealTimeProvider, ms_to_msk

LOGGER_NAME = "bot_a.paper.rest"


class RestPaperService:
    """Paper-сессия на реальном стакане T-Invest (REST-поллинг).

    Возвращает итоговый снапшот ядра, список сделок и траекторию для графиков.
    """

    def __init__(
        self,
        config: AppConfig,
        preset: PairPreset,
        client: TInvestRestClient,
        params: Optional[StrategyParams] = None,
        topology: Optional[ExecutionTopology] = None,
        interval_ms: Optional[int] = None,
        depth: int = 20,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._preset = preset
        self._client = client
        base_params = params or config.strategy
        self._params = replace(
            base_params,
            beta=float(preset.beta),
            beta_mode=(
                preset.beta_mode if base_params.beta_mode in ("", "auto") else base_params.beta_mode
            ),
            stat_sample_interval_seconds=float(config.execution.stat_sample_interval_seconds),
            use_entry_persistence=bool(config.execution.l2_entry_persistence),
            entry_confirm_samples=int(config.execution.l2_entry_confirm_samples),
        )
        self._topology = topology or config.execution.topology
        self._interval_ms = max(int(interval_ms or 200), 50)
        self._depth = max(int(depth), 1)
        self._log = logger or logging.getLogger(LOGGER_NAME)
        self._seen_trades: set = set()

    async def run(
        self,
        duration_seconds: Optional[float] = None,
        stop_event: Optional[asyncio.Event] = None,
        trace: Optional[Any] = None,
        warmup_days: Optional[int] = None,
        progress_seconds: float = 60.0,
    ) -> Tuple[Dict[str, Any], List[Any], Dict[str, Any]]:
        clock = RealTimeProvider()
        legs = [self._preset.leg1, self._preset.leg2]
        fee_calculator = TinkoffFeeCalculator(self._config.fees.tariff, clock=clock)
        gateway = SimulatedOrderExecutionGateway(
            metadata=legs,
            clock=clock,
            fee_calculator=fee_calculator,
            latency_ms=self._config.execution.simulated_order_latency_ms,
            queue_fill_factor=self._config.execution.queue_fill_factor,
            slippage_steps=self._config.execution.slippage_steps,
            order_book_depth=self._config.execution.order_book_depth,
            initial_cash_rub=self._config.execution.paper_initial_cash_rub,
        )
        guard = create_guard(self._config, legs)
        ledger = PortfolioLedger(self._config.execution.paper_initial_cash_rub)
        engine = create_engine(
            legs,
            self._params,
            self._topology,
            gateway,
            fee_calculator,
            clock,
            guard,
            ledger,
            self._log,
            daily_state_path=str(
                self._config.path(self._config.storage.state_dir)
                / f"daily_state_{self._preset.name}.json"
            ),
        )

        warmup: Dict[str, Any] = {}
        days = int(self._config.execution.bootstrap_warmup_days if warmup_days is None else warmup_days)
        if days > 0:
            from bot_a.services.bootstrap import bootstrap_warmup

            try:
                report = await bootstrap_warmup(
                    self._config, self._preset, engine, days=days, logger=self._log
                )
                warmup = report.to_dict()
            except Exception as exc:  # прогрев не должен блокировать сессию
                self._log.warning("Прогрев не удался (%s): %s", type(exc).__name__, exc)
                engine.set_bootstrap(False)

        await gateway.start()
        self._log.info(
            "Paper на живом стакане T-Invest (REST): %s, интервал %d мс, глубины %d, тариф %s",
            self._preset.name,
            self._interval_ms,
            self._depth,
            self._config.fees.tariff.value,
        )
        deadline = (time.monotonic() + duration_seconds) if duration_seconds else None
        started = time.monotonic()
        last_progress = started
        last_trades_poll = 0.0
        books = 0
        errors = 0
        last_trade_id = 0
        fresh: Dict[int, Any] = {}
        fetched_at: Dict[int, float] = {}
        failed_legs: List[int] = []
        syncer = PairBookSyncer(
            interval_ms=self._interval_ms,
            skew_factor=float(self._config.execution.l2_skew_factor),
        )

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    self._log.info("Остановка по сигналу")
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    self._log.info("Достигнута заданная длительность paper-сессии")
                    break
                for leg, meta in ((1, legs[0]), (2, legs[1])):
                    try:
                        book = await self._client.get_order_book(meta.instrument_uid, self._depth)
                    except TInvestRestError as exc:
                        errors += 1
                        failed_legs.append(leg)
                        if errors % 20 == 1:
                            self._log.warning("GetOrderBook %s: %s", meta.ticker, exc)
                        continue
                    except Exception as exc:  # noqa: BLE001 — сеть не должна ронять сессию
                        errors += 1
                        failed_legs.append(leg)
                        if errors % 20 == 1:
                            self._log.warning(
                                "GetOrderBook %s: %s: %s", meta.ticker, type(exc).__name__, exc
                            )
                        continue
                    if book is None:
                        failed_legs.append(leg)
                        continue
                    fresh[leg] = book
                    fetched_at[leg] = time.monotonic() * 1000.0
                decision = syncer.check(fresh, fetched_at, failed_legs)
                if decision.ok:
                    for leg in (1, 2):
                        await engine.on_order_book(leg, fresh[leg])
                        books += 1
                elif syncer.skew_skips % 50 == 1:
                    self._log.warning(
                        "Рассинхрон стаканов (%d): %s — тик пропущен для Z",
                        syncer.skew_skips,
                        decision.reason,
                    )
                fresh.clear()
                fetched_at.clear()
                failed_legs.clear()
                now = time.monotonic()
                if now - last_trades_poll >= 1.0:
                    last_trades_poll = now
                    await self._poll_trades(engine, legs)
                await engine.on_clock()
                if trace is not None:
                    trace.record(engine.metrics, engine.state.value)
                last_trade_id = self._log_new_trades(ledger, last_trade_id)
                if now - last_progress >= progress_seconds:
                    last_progress = now
                    self._log_progress(engine, ledger, started, books)
                await asyncio.sleep(self._interval_ms / 1000.0)
        except KeyboardInterrupt:
            self._log.warning("Прервано пользователем")
        finally:
            await engine.shutdown(flatten=self._config.risk.flatten_on_shutdown)
            for _ in range(6):
                if engine.position is None:
                    break
                await asyncio.sleep(1.0)
                await engine.on_clock()
            await gateway.stop()

        snapshot = engine.snapshot()
        snapshot["books"] = books
        snapshot["restErrors"] = errors
        snapshot["skewSkips"] = syncer.skew_skips
        snapshot["warmup"] = warmup
        snapshot["sessionSeconds"] = round(time.monotonic() - started, 1)
        return snapshot, list(ledger.trades), warmup

    async def _poll_trades(self, engine: Any, legs: List[Any]) -> None:
        # Окно 60 с + дедупликация: REST отдаёт принты с задержкой.
        since = datetime.now(tz=timezone.utc) - timedelta(seconds=60)
        for leg, meta in ((1, legs[0]), (2, legs[1])):
            try:
                trades = await self._client.get_last_trades(meta.instrument_uid, since)
            except TInvestRestError:
                continue
            for trade in trades:
                key = (meta.instrument_uid, trade.time_ms, str(trade.price), trade.quantity)
                if key in self._seen_trades:
                    continue
                self._seen_trades.add(key)
                await engine.on_trade(leg, trade)
        if len(self._seen_trades) > 50_000:
            self._seen_trades.clear()

    def _log_new_trades(self, ledger: PortfolioLedger, last_id: int) -> int:
        for trade in ledger.trades:
            if trade.trade_id <= last_id:
                continue
            last_id = trade.trade_id
            self._log.info(
                "СДЕЛКА #%d: %s 1 лот | вход %s MSK @ %s | выход %s MSK @ %s | %s | "
                "Z %.2f -> %.2f | удержание %.0f мин | gross %+.3f | fees %.3f | NET %+.3f RUB",
                trade.trade_id,
                trade.side_leg1.value.upper(),
                ms_to_msk(trade.opened_ms).strftime("%d.%m %H:%M:%S"),
                trade.leg1_entry_price,
                ms_to_msk(trade.closed_ms).strftime("%d.%m %H:%M:%S"),
                trade.leg1_exit_price,
                trade.exit_reason.value,
                trade.entry_z,
                trade.exit_z,
                trade.hold_seconds / 60.0,
                trade.gross_pnl_rub,
                trade.fees_rub,
                trade.net_pnl_rub,
            )
        return last_id

    def _log_progress(self, engine: Any, ledger: PortfolioLedger, started: float, books: int) -> None:
        snapshot = engine.snapshot()
        metrics = engine.metrics
        totals = ledger.totals()
        self._log.info(
            "%.0f c работы | %s | Z=%.2f | β=%.4f | сделок %d | net %s RUB | стаканов %d",
            time.monotonic() - started,
            snapshot["state"],
            snapshot["zscore"],
            metrics.beta if metrics else 0.0,
            totals.trades,
            totals.net_pnl_rub,
            books,
        )
