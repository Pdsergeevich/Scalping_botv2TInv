"""Живой/бумажный воркер стратегии v3.0: стакан -> минутные бары -> ФСМ пробоя.

Воркер ничего не знает о паре: он ведёт **один** инструмент. Источник котировок —
`MarketDataService.GetOrderBook` (REST/JSON или gRPC-клиент с тем же интерфейсом),
поэтому запускается и в `--mode paper`, и в `--mode watch` (watch = сигналы без сделок).

Поток данных:

1. поллинг стакана раз в `poll_ms`; из `microprice` строится текущий незавершённый бар;
2. при смене минуты бар считается завершённым и уходит в `MomentumBreakoutStateMachine`;
3. ФСМ возвращает событие входа/выхода, воркер исполняет его симулятором
   (цена ± `slippageSteps`, комиссия `TinkoffFeeCalculator`, ГО проверяется);
4. всё пишется в JSONL-журнал и итоговую сводку.

Размер позиции — как в бэктестере (`BreakoutBacktester.size`), чтобы paper и backtest
совпадали по деньгам.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from bot_a.backtest.breakout_backtest import BreakoutBacktester, BreakoutSettings, BreakoutTrade
from bot_a.economics.fees import Tariff, TinkoffFeeCalculator
from bot_a.engine.multiregime_fsm import regime_machine
from bot_a.engine.breakout_fsm import (
    Bar,
    BreakoutEvent,
    BreakoutExitReason,
    BreakoutParams,
    BreakoutSide,
    MomentumBreakoutStateMachine,
)
from bot_a.models.domain import InstrumentMetadata, OrderBookSnapshot
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import MSK

ZERO = Decimal(0)


@dataclass(slots=True)
class BreakoutBarBuilder:
    """Склеивает поток `microprice` в бары по `minutes` минут (MSK)."""

    minutes: int = 1
    current: Optional[Bar] = None

    @property
    def width_ms(self) -> int:
        return max(int(self.minutes), 1) * 60_000

    def update(self, microprice: Decimal, volume: int, time_ms: int) -> Optional[Bar]:
        """Возвращает **завершённый** бар, когда интервал сменился."""
        width = self.width_ms
        bucket = (time_ms // width) * width
        completed: Optional[Bar] = None
        if self.current is None:
            self.current = Bar(
                time_ms=bucket,
                open=microprice,
                high=microprice,
                low=microprice,
                close=microprice,
                volume=max(int(volume), 0),
            )
            return None
        bar = self.current
        if bucket > bar.time_ms:
            completed = bar
            self.current = Bar(
                time_ms=bucket,
                open=microprice,
                high=microprice,
                low=microprice,
                close=microprice,
                volume=max(int(volume), 0),
            )
            return completed
        bar.high = max(bar.high, microprice)
        bar.low = min(bar.low, microprice)
        bar.close = microprice
        bar.volume += max(int(volume), 0)
        return None


@dataclass(slots=True)
class BreakoutPaperState:
    """Текущее состояние бумажной торговли для дашборда/отчёта."""

    ticker: str
    started_ms: int
    position_side: Optional[str] = None
    quantity: int = 0
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    last_price: Optional[float] = None
    unrealized_rub: float = 0.0
    realized_rub: float = 0.0
    fees_rub: float = 0.0
    bars: int = 0
    trades: int = 0
    counters: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "startedAt": datetime.fromtimestamp(self.started_ms / 1000, tz=MSK).isoformat(
                timespec="seconds"
            ),
            "position": self.position_side,
            "quantity": self.quantity,
            "entryPrice": self.entry_price,
            "stopPrice": self.stop_price,
            "lastPrice": self.last_price,
            "unrealizedRub": round(self.unrealized_rub, 2),
            "realizedRub": round(self.realized_rub, 2),
            "feesRub": round(self.fees_rub, 2),
            "netRub": round(self.realized_rub + self.unrealized_rub, 2),
            "bars": self.bars,
            "trades": self.trades,
            "counters": dict(self.counters),
        }


def imbalance(book: OrderBookSnapshot, depth: int = 5, decay: float = 0.5) -> float:
    bid = 0.0
    ask = 0.0
    for index, level in enumerate(book.bids[:depth]):
        bid += float(level.quantity) * pow(2.718281828, -decay * index)
    for index, level in enumerate(book.asks[:depth]):
        ask += float(level.quantity) * pow(2.718281828, -decay * index)
    total = bid + ask
    return (bid - ask) / total if total > 0 else 0.0


def microprice(book: OrderBookSnapshot, depth: int = 5, decay: float = 0.5) -> Optional[Decimal]:
    if not book.has_two_sided_book:
        return book.last_price
    bid_volume = sum(
        float(level.quantity) * pow(2.718281828, -decay * index)
        for index, level in enumerate(book.bids[:depth])
    )
    ask_volume = sum(
        float(level.quantity) * pow(2.718281828, -decay * index)
        for index, level in enumerate(book.asks[:depth])
    )
    total = bid_volume + ask_volume
    if total <= 0:
        return (book.best_bid + book.best_ask) / 2 if book.has_two_sided_book else book.last_price
    price = (bid_volume * float(book.best_ask) + ask_volume * float(book.best_bid)) / total
    return dec(price)


class BreakoutLiveWorker:
    """Исполняет стратегию v3.0 на живом стакане (paper/watch)."""

    def __init__(
        self,
        meta: InstrumentMetadata,
        params: Optional[BreakoutParams] = None,
        settings: Optional[BreakoutSettings] = None,
        *,
        dry_run: bool = True,
        poll_ms: int = 1000,
        journal_path: Optional[Path] = None,
        logger: Optional[Any] = None,
        tariff: Tariff = Tariff.PREMIUM,
        bar_minutes: int = 1,
    ) -> None:
        self._meta = meta
        self._p = params or BreakoutParams()
        self._settings = settings or BreakoutSettings()
        self._dry_run = dry_run
        self._poll_ms = max(int(poll_ms), 200)
        self._journal = Path(journal_path) if journal_path else None
        self._log = logger
        self._backtester = BreakoutBacktester(meta, self._p, self._settings)
        self._fees = TinkoffFeeCalculator(tariff)
        self._machine: Optional[MomentumBreakoutStateMachine] = None
        self._bars = BreakoutBarBuilder(minutes=max(int(bar_minutes), 1))
        self._obi_samples: List[float] = []
        self.state = BreakoutPaperState(ticker=meta.ticker, started_ms=0)
        self.trades: List[BreakoutTrade] = []
        self._pending: Optional[tuple] = None
        self._last_dirty_volume = 0

    # ------------------------------------------------------------------ прогрев
    def warmup(self, candles: List[Bar]) -> None:
        """Прогоняет исторические бары через ФСМ без исполнения (индикаторы греются)."""
        self._machine = regime_machine(
            self._meta, self._p, fee_calculator=self._backtester._round_trip_fee  # noqa: SLF001
        )
        for bar in candles:
            self._machine.on_bar(bar)
        self.state.bars = len(candles)
        if self._log is not None and candles:
            self._log.info(
                "Пробой %s: прогрев %s минутных баров, индикаторы готовы",
                self._meta.ticker,
                len(candles),
            )

    # ------------------------------------------------------------------ основной цикл
    async def run(
        self,
        client: Any,
        duration_seconds: Optional[float] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        if self._machine is None:
            self.warmup([])
        loop = asyncio.get_running_loop()
        deadline = loop.time() + duration_seconds if duration_seconds else None
        self.state.started_ms = int(datetime.now(tz=MSK).timestamp() * 1000)
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if deadline is not None and loop.time() >= deadline:
                break
            try:
                book = await client.get_order_book(self._meta.instrument_uid, depth=5)
            except Exception as exc:  # noqa: BLE001 — сеть не должна убивать сессию
                if self._log is not None:
                    self._log.warning("Пробой %s: ошибка стакана: %s", self._meta.ticker, exc)
                await asyncio.sleep(self._poll_ms / 1000.0)
                continue
            if book is not None:
                self.on_book(book)
            await asyncio.sleep(self._poll_ms / 1000.0)
        summary = self.summary()
        if self._log is not None:
            self._log.info("Пробой %s: итог %s", self._meta.ticker, json.dumps(summary["net"]))
        return summary

    def on_book(self, book: OrderBookSnapshot) -> None:
        price = microprice(book)
        if price is None or price <= 0:
            return
        obi = imbalance(book)
        self._obi_samples.append(obi)
        if len(self._obi_samples) > 60:
            self._obi_samples.pop(0)
        volume = int(book.bid_volume + book.ask_volume)
        delta_volume = max(volume - self._last_dirty_volume, 0)
        self._last_dirty_volume = volume
        completed = self._bars.update(price, delta_volume, book.time_ms)
        self.state.last_price = float(price)
        if completed is None:
            return
        self.state.bars += 1
        obi_avg = sum(self._obi_samples) / len(self._obi_samples)
        for event in self._machine.on_bar(completed, obi=obi_avg):
            self._handle_events(event)
        self._mark_to_market(price)

    # ------------------------------------------------------------------ исполнение
    def _handle_events(self, event: BreakoutEvent) -> None:
        if event.kind == "entry":
            self._open(event)
        else:
            self._close(event)

    def _open(self, event: BreakoutEvent) -> None:
        position = self._machine.position
        if position is None:
            return
        stop_points = abs(float(position.entry_price - position.initial_stop))
        quantity = self._backtester.size(stop_points)
        if quantity <= 0:
            self._machine.position = None
            self.state.counters["noCapitalSkips"] = (
                self.state.counters.get("noCapitalSkips", 0) + 1
            )
            return
        fill, stop = self._backtester._apply_entry_fill(  # noqa: SLF001
            event.price, event.side, stop_points
        )
        position.quantity = quantity
        position.entry_price = fill
        position.stop_price = stop
        position.initial_stop = stop
        position.fee_round_trip_rub = self._backtester._round_trip_fee(  # noqa: SLF001
            fill, quantity
        )
        event.price = fill
        entry_fee = self._fees.register_notional(
            self._meta.value_in_rub(fill) * Decimal(quantity)
        )
        self._pending = (event, quantity, entry_fee, position.atr_points, position.entry_ms)
        self.state.position_side = event.side.value
        self.state.quantity = quantity
        self.state.entry_price = float(fill)
        self.state.stop_price = float(stop)
        self._journal_write({"event": "entry", **_event_payload(event), "quantity": quantity})
        if self._log is not None:
            self._log.info(
                "Пробой %s: ВХОД %s %s контрактов по %s (стоп %s, ATR %s) — %s",
                self._meta.ticker,
                event.side.value,
                quantity,
                fill,
                stop,
                round(position.atr_points, 4),
                "watch" if self._dry_run else "paper",
            )

    def _close(self, event: BreakoutEvent) -> None:
        if self._pending is None:
            return
        entry_event, quantity, entry_fee, atr_points, entry_ms = self._pending
        self._pending = None
        exit_price = self._backtester._apply_exit_fill(  # noqa: SLF001
            event.price, event.side
        )
        exit_fee = self._fees.register_notional(
            self._meta.value_in_rub(exit_price) * Decimal(quantity)
        )
        sign = Decimal(1) if event.side is BreakoutSide.LONG else Decimal(-1)
        gross = float(
            (exit_price - entry_event.price)
            * sign
            * self._meta.rub_per_price_unit
            * Decimal(quantity)
        )
        fees = float(entry_fee + exit_fee)
        trade = BreakoutTrade(
            ticker=self._meta.ticker,
            side=event.side,
            quantity=quantity,
            entry_ms=entry_event.time_ms,
            exit_ms=event.time_ms,
            entry_price=entry_event.price,
            exit_price=exit_price,
            gross_rub=gross,
            fees_rub=fees,
            net_rub=gross - fees,
            reason=event.reason or BreakoutExitReason.DATA_END,
            atr_points=atr_points,
            mfe_points=0.0,
            hold_minutes=(event.time_ms - entry_ms) / 60_000.0,
        )
        self.trades.append(trade)
        self.state.realized_rub += trade.net_rub
        self.state.fees_rub += trade.fees_rub
        self.state.trades += 1
        self.state.position_side = None
        self.state.quantity = 0
        self.state.entry_price = None
        self.state.stop_price = None
        self._journal_write(
            {
                "event": "exit",
                **_event_payload(event),
                "quantity": quantity,
                "netRub": round(trade.net_rub, 2),
            }
        )
        if self._log is not None:
            self._log.info(
                "Пробой %s: ВЫХОД %s по %s (%s) нетто %+.2f RUB",
                self._meta.ticker,
                event.side.value,
                exit_price,
                trade.reason.value,
                trade.net_rub,
            )

    def _mark_to_market(self, price: Decimal) -> None:
        position = self._machine.position if self._machine else None
        if position is None:
            self.state.unrealized_rub = 0.0
            self.state.stop_price = None
            return
        points = position.unrealized_points(price)
        self.state.unrealized_rub = (
            points * float(self._meta.rub_per_price_unit) * position.quantity
        )
        self.state.stop_price = float(position.stop_price)

    # ------------------------------------------------------------------ вывод
    def summary(self) -> Dict[str, Any]:
        payload = self.state.to_dict()
        payload["trades"] = self.state.trades
        payload["tradeList"] = [trade.to_dict() for trade in self.trades]
        payload["net"] = {
            "realizedRub": round(self.state.realized_rub, 2),
            "unrealizedRub": round(self.state.unrealized_rub, 2),
            "feesRub": round(self.state.fees_rub, 2),
            "trades": self.state.trades,
        }
        return payload

    def _journal_write(self, row: Dict[str, Any]) -> None:
        if self._journal is None:
            return
        self._journal.parent.mkdir(parents=True, exist_ok=True)
        with self._journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _event_payload(event: BreakoutEvent) -> Dict[str, Any]:
    return {
        "time": datetime.fromtimestamp(event.time_ms / 1000, tz=MSK).isoformat(timespec="seconds"),
        "side": event.side.value,
        "price": float(event.price),
        "reason": event.reason.value if event.reason else None,
        "note": event.note,
    }
