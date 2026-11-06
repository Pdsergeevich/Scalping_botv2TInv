"""Мульти-тикерный свинг-раннер v3.2 (`--strategy breakout --ticker ALL_SWING`).

Один процесс ведёт **корзину** инструментов с общим счётом:

* размер считается от общего капитала (`riskPct × equity`), а не от «мини-счёта» на тикер;
* ГО всех открытых позиций учитывается совместно с лимитом утилизации;
* каждый инструмент имеет собственный бар-билдер, ФСМ и стакан, но касса — одна;
* открытые позиции и капитал сохраняются в `state/swing_positions.json`, поэтому
  перенос через ночь переживает перезапуск процесса (после рестарта позиция
  восстанавливается вместе со стопом, MFE и включённым трейлом).

Раннер не зависит от источника котировок: подходит и `TInvestRestClient`, и `IssQuoteSource`.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from bot_a.backtest.breakout_backtest import BreakoutTrade
from bot_a.backtest.fills import apply_entry_fill, apply_exit_fill, notional_rub
from bot_a.economics.fees import Tariff, TinkoffFeeCalculator
from bot_a.engine.multiregime_fsm import regime_machine
from bot_a.engine.breakout_fsm import (
    Bar,
    BreakoutEvent,
    BreakoutExitReason,
    BreakoutParams,
    BreakoutPosition,
    BreakoutSide,
    MomentumBreakoutStateMachine,
)
from bot_a.models.domain import InstrumentMetadata, OrderBookSnapshot
from bot_a.services.breakout_worker import BreakoutBarBuilder, imbalance, microprice
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import MSK

ZERO = Decimal(0)


@dataclass(slots=True)
class SwingBasketSettings:
    """Настройки корзины."""

    capital_rub: float = 50_000.0
    risk_pct_per_trade: float = 1.0
    margin_safety_buffer_pct: float = 25.0
    max_margin_utilization_pct: float = 60.0
    max_concurrent_positions: int = 0
    # v4.3: безрисковые «раннеры» (стоп выше входа) не занимают риск-слот.
    free_slot_on_breakeven: bool = True
    max_total_open_positions: int = 8
    # v4.4: доля бюджета ГО на одну позицию (тяжёлый контракт не съедает весь лимит).
    max_position_margin_pct: float = 35.0
    bar_minutes: int = 30
    poll_ms: int = 2000
    state_path: Optional[Path] = None
    journal_path: Optional[Path] = None
    tariff: Tariff = Tariff.PREMIUM
    dry_run: bool = True


@dataclass(slots=True)
class BasketPosition:
    """Позиция корзины (то, что переживает перезапуск)."""

    ticker: str
    side: BreakoutSide
    quantity: int
    entry_price: Decimal
    entry_ms: int
    stop_price: Decimal
    initial_stop: Decimal
    atr_points: float
    margin_rub: float
    entry_fee: Decimal
    mfe_points: float = 0.0
    max_price: Decimal = ZERO
    min_price: Decimal = ZERO
    breakeven_done: bool = False
    soft_stop_done: bool = False
    scale_out_done: bool = False
    trail_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "side": self.side.value,
            "quantity": self.quantity,
            "entryPrice": str(self.entry_price),
            "entryTime": datetime.fromtimestamp(self.entry_ms / 1000, tz=MSK).isoformat(
                timespec="seconds"
            ),
            "entryMs": self.entry_ms,
            "stopPrice": str(self.stop_price),
            "initialStop": str(self.initial_stop),
            "atrPoints": round(self.atr_points, 6),
            "marginRub": round(self.margin_rub, 2),
            "entryFee": str(self.entry_fee),
            "mfePoints": round(self.mfe_points, 6),
            "maxPrice": str(self.max_price),
            "minPrice": str(self.min_price),
            "breakevenDone": self.breakeven_done,
            "softStopDone": self.soft_stop_done,
            "scaleOutDone": self.scale_out_done,
            "trailActive": self.trail_active,
        }

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "BasketPosition":
        return BasketPosition(
            ticker=str(raw["ticker"]),
            side=BreakoutSide(raw["side"]),
            quantity=int(raw["quantity"]),
            entry_price=dec(raw["entryPrice"]),
            entry_ms=int(raw.get("entryMs") or 0),
            stop_price=dec(raw["stopPrice"]),
            initial_stop=dec(raw.get("initialStop") or raw["stopPrice"]),
            atr_points=float(raw.get("atrPoints") or 0.0),
            margin_rub=float(raw.get("marginRub") or 0.0),
            entry_fee=dec(raw.get("entryFee") or 0),
            mfe_points=float(raw.get("mfePoints") or 0.0),
            max_price=dec(raw.get("maxPrice") or 0),
            min_price=dec(raw.get("minPrice") or 0),
            breakeven_done=bool(raw.get("breakevenDone")),
            soft_stop_done=bool(raw.get("softStopDone")),
            scale_out_done=bool(raw.get("scaleOutDone")),
            trail_active=bool(raw.get("trailActive")),
        )


class SwingBasketRunner:
    """Общий счёт на несколько инструментов: стакан → бары → ФСМ → исполнение."""

    def __init__(
        self,
        instruments: Sequence[Tuple[InstrumentMetadata, BreakoutParams]],
        settings: Optional[SwingBasketSettings] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self._settings = settings or SwingBasketSettings()
        self._log = logger
        self._fees = TinkoffFeeCalculator(self._settings.tariff)
        self._metas: Dict[str, InstrumentMetadata] = {}
        self._params: Dict[str, BreakoutParams] = {}
        self._machines: Dict[str, MomentumBreakoutStateMachine] = {}
        self._builders: Dict[str, BreakoutBarBuilder] = {}
        self._positions: Dict[str, BasketPosition] = {}
        self._obi: Dict[str, List[float]] = {}
        self._equity = float(self._settings.capital_rub)
        self._realized = 0.0
        self._fees_paid = 0.0
        self._trades: List[BreakoutTrade] = []
        self._started_ms = int(datetime.now(tz=MSK).timestamp() * 1000)
        self._last_ms: Dict[str, int] = {}
        for meta, params in instruments:
            self._register(meta, params)

    # ------------------------------------------------------------------ регистрация
    def _register(self, meta: InstrumentMetadata, params: BreakoutParams) -> None:
        ticker = meta.ticker
        self._metas[ticker] = meta
        self._params[ticker] = params
        self._machines[ticker] = regime_machine(
            meta,
            params,
            fee_calculator=lambda price, qty, _meta=meta: self._round_trip_fee(_meta, price, qty),
        )
        self._builders[ticker] = BreakoutBarBuilder(minutes=max(int(self._settings.bar_minutes), 1))
        self._obi[ticker] = []

    @property
    def tickers(self) -> List[str]:
        return sorted(self._metas)

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def margin_used(self) -> float:
        return sum(position.margin_rub for position in self._positions.values())

    @property
    def trades(self) -> List[BreakoutTrade]:
        return list(self._trades)

    # ------------------------------------------------------------------ прогрев
    def warmup(self, candles: Dict[str, Sequence[Bar]]) -> None:
        """Проигрывает историю через ФСМ, чтобы индикаторы были готовы к live-барам."""
        for ticker, bars in candles.items():
            machine = self._machines.get(ticker)
            if machine is None:
                continue
            for bar in bars:
                machine.on_bar(bar)
            machine.position = None  # история нужна только индикаторам
            if self._log is not None:
                self._log.info(
                    "Свинг-корзина %s: прогрев %d баров (%d мин)",
                    ticker,
                    len(bars),
                    self._settings.bar_minutes,
                )

    # ------------------------------------------------------------------ состояние
    def restore(self) -> bool:
        """Восстанавливает позиции и капитал из `state/swing_positions.json`."""
        path = self._settings.state_path
        if path is None or not Path(path).exists():
            return False
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if self._log is not None:
                self._log.warning("Не удалось прочитать состояние свинга: %s", exc)
            return False
        self._equity = float(raw.get("equity") or self._settings.capital_rub)
        self._realized = float(raw.get("realizedRub") or 0.0)
        self._fees_paid = float(raw.get("feesRub") or 0.0)
        for ticker, payload in (raw.get("positions") or {}).items():
            if ticker not in self._machines:
                continue
            position = BasketPosition.from_dict(payload)
            self._positions[ticker] = position
            self._machines[ticker].position = BreakoutPosition(
                side=position.side,
                quantity=position.quantity,
                entry_price=position.entry_price,
                entry_ms=position.entry_ms,
                atr_points=position.atr_points,
                stop_price=position.stop_price,
                initial_stop=position.initial_stop,
                fee_round_trip_rub=self._round_trip_fee(
                    self._metas[ticker], position.entry_price, position.quantity
                ),
                mfe_points=position.mfe_points,
                max_price=position.max_price,
                min_price=position.min_price,
                breakeven_done=position.breakeven_done,
                soft_stop_done=position.soft_stop_done,
                scale_out_done=position.scale_out_done,
                trail_active=position.trail_active,
            )
            if self._log is not None:
                self._log.info(
                    "Свинг-корзина: восстановлена позиция %s %s x%d по %s (стоп %s)",
                    ticker,
                    position.side.value,
                    position.quantity,
                    position.entry_price,
                    position.stop_price,
                )
        return bool(self._positions)

    def save_state(self) -> None:
        path = self._settings.state_path
        if path is None:
            return
        payload = {
            "updatedAt": datetime.now(tz=MSK).isoformat(timespec="seconds"),
            "capital": round(self._settings.capital_rub, 2),
            "equity": round(self._equity, 2),
            "realizedRub": round(self._realized, 2),
            "feesRub": round(self._fees_paid, 2),
            "marginUsed": round(self.margin_used, 2),
            "positions": {ticker: pos.to_dict() for ticker, pos in self._positions.items()},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ основной цикл
    async def run(
        self,
        clients: Dict[str, Any],
        duration_seconds: Optional[float] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + duration_seconds if duration_seconds else None
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if deadline is not None and loop.time() >= deadline:
                break
            await asyncio.gather(
                *(self._poll(ticker, client) for ticker, client in clients.items()),
                return_exceptions=True,
            )
            await asyncio.sleep(self._settings.poll_ms / 1000.0)
        self.save_state()
        return self.summary()

    async def _poll(self, ticker: str, client: Any) -> None:
        meta = self._metas[ticker]
        try:
            book = await client.get_order_book(meta.instrument_uid, depth=5)
        except Exception as exc:  # noqa: BLE001 — сеть не должна ронять корзину
            if self._log is not None:
                self._log.warning("Свинг-корзина %s: ошибка стакана: %s", ticker, exc)
            return
        if book is not None:
            self.on_book(ticker, book)

    # ------------------------------------------------------------------ обработка стакана
    def on_book(self, ticker: str, book: OrderBookSnapshot) -> Optional[BreakoutEvent]:
        price = microprice(book)
        if price is None or price <= 0:
            return None
        samples = self._obi[ticker]
        samples.append(imbalance(book))
        if len(samples) > 60:
            samples.pop(0)
        volume = int(book.bid_volume + book.ask_volume)
        previous = self._last_ms.get(ticker, 0)
        delta = max(volume - previous, 0) if previous else 0
        if volume and not previous:
            delta = 0
        self._last_ms[ticker] = volume
        completed = self._builders[ticker].update(price, delta, book.time_ms)
        if completed is None:
            return None
        obi_avg = sum(samples) / len(samples)
        event: Optional[BreakoutEvent] = None
        for item in self._machines[ticker].on_bar(completed, obi=obi_avg):
            if item.kind == "entry":
                self._open(ticker, item)
                event = item
            elif item.kind == "partial":
                self._partial(ticker, item)
            else:
                self._close(ticker, item)
                event = item
        if event is not None:
            self.save_state()
        return event

    # ------------------------------------------------------------------ исполнение
    def _open(self, ticker: str, event: BreakoutEvent) -> None:
        machine = self._machines[ticker]
        meta = self._metas[ticker]
        position = machine.position
        if position is None:
            return
        if self._settings.max_concurrent_positions > 0:
            at_risk = len(self._positions)
            if self._settings.free_slot_on_breakeven:
                at_risk = sum(
                    1
                    for ticker_open in self._positions
                    if not getattr(self._machines[ticker_open].position, "breakeven_done", False)
                )
            if at_risk >= self._settings.max_concurrent_positions:
                machine.position = None
                return
        if (
            self._settings.max_total_open_positions > 0
            and len(self._positions) >= self._settings.max_total_open_positions
        ):
            machine.position = None
            return
        stop_points = abs(float(position.entry_price - position.initial_stop))
        quantity = self.size(meta, stop_points)
        if quantity <= 0:
            machine.position = None
            if self._log is not None:
                self._log.info(
                    "Свинг-корзина %s: вход пропущен — не хватает риска/ГО", ticker
                )
            return
        fill, stop = apply_entry_fill(meta, self._params[ticker], event.price, event.side, stop_points)
        position.quantity = quantity
        position.entry_price = fill
        position.stop_price = stop
        position.initial_stop = stop
        position.fee_round_trip_rub = self._round_trip_fee(meta, fill, quantity)
        event.price = fill
        entry_fee = self._fees.register_notional(notional_rub(meta, fill, quantity), ticker)
        margin = float(meta.position_margin_rub(1)) * (
            1.0 + float(self._settings.margin_safety_buffer_pct) / 100.0
        ) * quantity
        self._fees_paid += float(entry_fee)
        self._positions[ticker] = BasketPosition(
            ticker=ticker,
            side=event.side,
            quantity=quantity,
            entry_price=fill,
            entry_ms=event.time_ms,
            stop_price=stop,
            initial_stop=stop,
            atr_points=position.atr_points,
            margin_rub=margin,
            entry_fee=entry_fee,
        )
        if self._log is not None:
            self._log.info(
                "Свинг-корзина %s: ВХОД %s %d x %s (стоп %s, ATR %s, ГО %.0f ₽)",
                ticker,
                event.side.value,
                quantity,
                fill,
                stop,
                round(position.atr_points, 4),
                margin,
            )

    def _partial(self, ticker: str, event: BreakoutEvent) -> None:
        position = self._positions.get(ticker)
        entry_event = self._machines[ticker].position
        if position is None or entry_event is None:
            return
        closed = max(min(int(event.quantity), position.quantity), 1)
        exit_price = apply_exit_fill(self._metas[ticker], self._params[ticker], event.price, event.side)
        exit_fee = self._fees.register_notional(
            notional_rub(self._metas[ticker], exit_price, closed), ticker
        )
        share_fee = position.entry_fee * Decimal(closed) / Decimal(max(position.quantity, 1))
        gross = self._gross(self._metas[ticker], position.entry_price, exit_price, event.side, closed)
        fees = float(share_fee + exit_fee)
        self._record(ticker, position, event, exit_price, closed, gross, fees)
        share_margin = position.margin_rub * closed / max(position.quantity, 1)
        position.quantity -= closed
        position.margin_rub -= share_margin
        position.entry_fee -= share_fee
        self._fees_paid += float(exit_fee)
        if self._log is not None:
            self._log.info(
                "Свинг-корзина %s: ЧАСТИЧНО закрыто %d, осталось %d",
                ticker,
                closed,
                position.quantity,
            )

    def _close(self, ticker: str, event: BreakoutEvent) -> None:
        position = self._positions.pop(ticker, None)
        if position is None:
            return
        exit_price = apply_exit_fill(self._metas[ticker], self._params[ticker], event.price, event.side)
        exit_fee = self._fees.register_notional(
            notional_rub(self._metas[ticker], exit_price, position.quantity), ticker
        )
        gross = self._gross(
            self._metas[ticker], position.entry_price, exit_price, event.side, position.quantity
        )
        fees = float(position.entry_fee + exit_fee)
        self._fees_paid += float(exit_fee)
        trade = self._record(
            ticker, position, event, exit_price, position.quantity, gross, fees
        )
        self._positions.pop(ticker, None)
        if self._log is not None:
            self._log.info(
                "Свинг-корзина %s: ВЫХОД %s по %s (%s) нетто %+.2f ₽",
                ticker,
                event.side.value,
                exit_price,
                trade.reason.value,
                trade.net_rub,
            )

    def _record(
        self,
        ticker: str,
        position: BasketPosition,
        event: BreakoutEvent,
        exit_price: Decimal,
        closed: int,
        gross: float,
        fees: float,
    ) -> BreakoutTrade:
        net = gross - fees
        trade = BreakoutTrade(
            ticker=ticker,
            side=event.side,
            quantity=closed,
            entry_ms=position.entry_ms,
            exit_ms=event.time_ms,
            entry_price=position.entry_price,
            exit_price=exit_price,
            gross_rub=gross,
            fees_rub=fees,
            net_rub=net,
            reason=event.reason or BreakoutExitReason.DATA_END,
            atr_points=position.atr_points,
            mfe_points=position.mfe_points,
            hold_minutes=(event.time_ms - position.entry_ms) / 60_000.0,
        )
        self._trades.append(trade)
        self._realized += net
        self._equity += net
        return trade

    # ------------------------------------------------------------------ расчёты
    def _round_trip_fee(self, meta: InstrumentMetadata, price: Decimal, quantity: int) -> Decimal:
        return self._fees.fee_for_notional(notional_rub(meta, price, quantity)) * 2

    @staticmethod
    def _gross(
        meta: InstrumentMetadata, entry: Decimal, exit_price: Decimal, side: BreakoutSide, quantity: int
    ) -> float:
        sign = Decimal(1) if side is BreakoutSide.LONG else Decimal(-1)
        return float((exit_price - entry) * sign * meta.rub_per_price_unit * Decimal(quantity))

    def size(self, meta: InstrumentMetadata, stop_points: float) -> int:
        """Контракты от общего капитала и свободного ГО (как в бэктестере)."""
        settings = self._settings
        risk_budget = max(self._equity, 0.0) * float(settings.risk_pct_per_trade) / 100.0
        stop_rub = max(stop_points * float(meta.rub_per_price_unit), 1e-9)
        by_risk = int(risk_budget // stop_rub)
        margin_per_contract = float(meta.position_margin_rub(1)) * (
            1.0 + float(settings.margin_safety_buffer_pct) / 100.0
        )
        util_cap = max(self._equity, 0.0) * max(float(settings.max_margin_utilization_pct), 1.0) / 100.0
        free = max(min(util_cap, self._equity) - self.margin_used, 0.0)
        by_margin = int(free // margin_per_contract) if margin_per_contract > 0 else by_risk
        quantity = min(by_risk, by_margin)
        if settings.max_position_margin_pct > 0 and margin_per_contract > 0:
            per_position = util_cap * float(settings.max_position_margin_pct) / 100.0
            quantity = min(quantity, int(per_position // margin_per_contract))
        return max(quantity, 0)

    # ------------------------------------------------------------------ отчёт
    def summary(self) -> Dict[str, Any]:
        return {
            "mode": "ALL_SWING",
            "dryRun": self._settings.dry_run,
            "startedAt": datetime.fromtimestamp(self._started_ms / 1000, tz=MSK).isoformat(
                timespec="seconds"
            ),
            "capital": round(self._settings.capital_rub, 2),
            "equity": round(self._equity, 2),
            "realizedRub": round(self._realized, 2),
            "feesRub": round(self._fees_paid, 2),
            "marginUsed": round(self.margin_used, 2),
            "trades": len(self._trades),
            "openPositions": {
                ticker: position.to_dict() for ticker, position in self._positions.items()
            },
            "tradeList": [
                {
                    "ticker": trade.ticker,
                    "side": trade.side.value,
                    "quantity": trade.quantity,
                    "entry": datetime.fromtimestamp(trade.entry_ms / 1000, tz=MSK).isoformat(
                        timespec="seconds"
                    ),
                    "exit": datetime.fromtimestamp(trade.exit_ms / 1000, tz=MSK).isoformat(
                        timespec="seconds"
                    ),
                    "netRub": round(trade.net_rub, 2),
                    "reason": trade.reason.value,
                }
                for trade in self._trades
            ],
        }


def basket_from_config(config: Any, tickers: Optional[Iterable[str]] = None) -> List[str]:
    """Список инструментов корзины: явный список → `swingBasket` из конфига → все пресеты."""
    # Регистр сохраняем: continuous-серии (`Eu_CONT`) привязаны к именам файлов кэша.
    if tickers:
        return [str(ticker) for ticker in tickers]
    configured = getattr(config, "swing_basket", None) or []
    if configured:
        return [str(ticker) for ticker in configured]
    return [name for name in config.breakout_names()]
