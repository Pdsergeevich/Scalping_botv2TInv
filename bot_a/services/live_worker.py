"""Живой торговый воркер: paper (симулятор) и live (реальные заявки)."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Any, Dict, Optional

from bot_a.broker import tinvest as broker
from bot_a.config import AppConfig, PairPreset
from bot_a.economics.fees import TinkoffFeeCalculator
from bot_a.engine.factory import create_engine, create_guard
from bot_a.engine.fsm import StrategyParams
from bot_a.engine.ledger import PortfolioLedger
from bot_a.execution.live_gateway import TinkoffLiveExecutionGateway
from bot_a.execution.simulated_gateway import SimulatedOrderExecutionGateway
from bot_a.models.domain import MarketEvent
from bot_a.utils.channel import Channel
from bot_a.utils.time_provider import RealTimeProvider

LOGGER_NAME = "bot_a.live"


class LiveTradingWorker:
    """Единый цикл для ``--mode paper`` и ``--mode live``."""

    def __init__(
        self,
        config: AppConfig,
        preset: PairPreset,
        client: Any,
        paper: bool,
        params: Optional[StrategyParams] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._preset = preset
        self._client = client
        self._paper = paper
        self._params = params or config.strategy
        self._log = logger or logging.getLogger(LOGGER_NAME)
        self._warmup_report: Dict[str, Any] = {}

    async def run(
        self,
        stop_event: Optional[asyncio.Event] = None,
        duration_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        stop = stop_event or asyncio.Event()
        mode = "paper (симулятор)" if self._paper else "LIVE (реальные заявки)"
        self._log.info(
            "Запуск режима %s: %s / %s, топология %s, тариф %s",
            mode,
            self._preset.leg1.ticker,
            self._preset.leg2.ticker,
            self._config.execution.topology.value,
            self._config.fees.tariff.value,
        )
        account_id = await self._resolve_account_id()
        legs = await self._resolve_legs()
        clock = RealTimeProvider()
        fee_calculator = TinkoffFeeCalculator(self._config.fees.tariff, clock=clock)
        guard = create_guard(self._config, legs)
        ledger = PortfolioLedger(self._config.execution.paper_initial_cash_rub)

        if self._paper:
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
        else:
            order_client = broker.TinkoffOrderClient(self._client, account_id, legs)
            gateway = TinkoffLiveExecutionGateway(
                metadata=legs,
                clock=clock,
                fee_calculator=fee_calculator,
                order_client=order_client,
                poll_interval_ms=self._config.execution.order_poll_interval_ms,
                cash_poll_interval_ms=self._config.execution.cash_poll_interval_ms,
            )

        engine = create_engine(
            legs,
            replace(
                self._params,
                beta=float(self._preset.beta),
                beta_mode=(
                    self._preset.beta_mode
                    if self._params.beta_mode in ("", "auto")
                    else self._params.beta_mode
                ),
                stat_sample_interval_seconds=float(
                    self._config.execution.stat_sample_interval_seconds
                ),
            ),
            self._config.execution.topology,
            gateway,
            fee_calculator,
            clock,
            guard,
            ledger,
            self._log,
            daily_state_path=str(self._daily_state_path()),
        )

        await self._bootstrap(engine, legs)

        channel: Channel[MarketEvent] = Channel(maxsize=50_000)
        stream = broker.TinkoffMarketDataStream(
            self._client,
            legs,
            depth=self._config.execution.order_book_depth,
            clock=clock,
        )

        async def pump() -> None:
            try:
                async with stream:
                    self._log.info("Подписка на стаканы/сделки активна")
                    async for event in stream.events():
                        if stop.is_set():
                            break
                        if not channel.try_write(event):
                            await channel.write(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # разрыв стрима — останавливаем торговлю
                self._log.error("Ошибка рыночного стрима: %s", exc)
                stop.set()

        await gateway.start()
        pump_task = asyncio.create_task(pump())
        deadline = (time.monotonic() + duration_seconds) if duration_seconds else None
        last_report = time.monotonic()
        try:
            while not stop.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    self._log.info("Достигнута заданная длительность работы")
                    break
                try:
                    event = await asyncio.wait_for(channel.read(), timeout=0.05)
                except asyncio.TimeoutError:
                    event = None
                if event is not None and not stop.is_set():
                    if event.is_book and event.book is not None:
                        await engine.on_order_book(event.leg, event.book)
                    elif event.trade is not None:
                        await engine.on_trade(event.leg, event.trade)
                await engine.on_clock()
                if time.monotonic() - last_report >= 30.0:
                    last_report = time.monotonic()
                    snapshot = engine.snapshot()
                    self._log.info(
                        "Статус: %s | Z=%.2f | сделок=%d | net=%s RUB | комиссии=%s RUB",
                        snapshot["state"],
                        snapshot["zscore"],
                        snapshot["trades"],
                        snapshot["netPnlRub"],
                        snapshot["feesRub"],
                    )
        finally:
            stop.set()
            pump_task.cancel()
            await asyncio.gather(pump_task, return_exceptions=True)
            await engine.shutdown(flatten=self._config.risk.flatten_on_shutdown)
            for _ in range(12):
                if engine.position is None:
                    break
                await asyncio.sleep(0.25)
                await engine.on_clock()
            await gateway.stop()
            self._log.info("Работа завершена. Итог: %s", engine.snapshot())
        snapshot = engine.snapshot()
        snapshot["accountId"] = account_id
        snapshot["mode"] = "paper" if self._paper else "live"
        snapshot["preset"] = self._preset.name
        snapshot["warmup"] = self._warmup_report
        return snapshot

    def _daily_state_path(self):
        """Файл персистентного состояния дня (v2.4.1), отдельный на пресет."""
        base = self._config.path(self._config.storage.state_dir)
        return base / f"daily_state_{self._preset.name}.json"

    async def _bootstrap(self, engine, legs) -> None:
        """Автоматический прогрев статистики из MOEX ISS перед live/paper сессией."""
        days = int(self._config.execution.bootstrap_warmup_days)
        if days <= 0:
            self._log.info("Прогрев отключён (bootstrapWarmupDays = 0)")
            return
        from bot_a.services.bootstrap import bootstrap_warmup

        try:
            report = await bootstrap_warmup(
                self._config,
                self._preset.with_metadata(legs[0], legs[1]),
                engine,
                days=days,
                logger=self._log,
            )
        except Exception as exc:  # прогрев не должен блокировать торговлю
            self._log.warning("Прогрев из MOEX ISS не удался (%s): %s", type(exc).__name__, exc)
            engine.set_bootstrap(False)
            return
        self._warmup_report = report.to_dict()

    async def _resolve_account_id(self) -> str:
        if self._config.t_invest.account_id:
            return self._config.t_invest.account_id
        accounts = await broker.get_accounts(self._client)
        for account in accounts:
            if "FULL" in account.access_level.upper():
                return account.account_id
        if accounts:
            return accounts[0].account_id
        raise RuntimeError("Не найдено ни одного счёта T-Invest")

    async def _resolve_legs(self):
        legs = []
        for index, meta in ((1, self._preset.leg1), (2, self._preset.leg2)):
            try:
                resolved = await broker.resolve_instrument(self._client, meta.ticker, meta.class_code, meta.market)
            except Exception as exc:
                raise RuntimeError(
                    f"Не удалось разрешить Leg{index} {meta.ticker} ({meta.class_code}): {exc}. "
                    "Проверьте пресет и выполните --mode preflight"
                ) from exc
            self._log.info(
                "Leg%d: %s uid=%s, шаг=%s, стоимость шага=%s, ГО=%s",
                index,
                resolved.ticker,
                resolved.instrument_uid,
                resolved.min_price_increment,
                resolved.min_price_increment_amount,
                resolved.position_margin_rub(1),
            )
            legs.append(resolved)
        return legs
