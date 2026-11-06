"""Портфельный движок v3.2: несколько инструментов с **общим счётом** и общей маржой.

Зачем: в v3.1 капитал делился на изолированные «мини-счёта» (50 000 / 3 = 16 667 ₽), и
`USDRUBF` с ГО 12 316 ₽ и стопом ≈273 ₽ не мог позволить себе ни одного контракта
(риск-бюджет 1.5 % от 16 667 ₽ = 250 ₽ < 273 ₽). Здесь инструменты торгуют с одного счёта:

* риск на сделку считается от **текущего капитала счёта** (`risk_pct × equity`);
* размер ограничен **свободным остатком под ГО всех открытых позиций сразу**
  (`free_cash = equity × maxMarginUtilization − Σ ГО_открытых × (1 + буфер)`);
* реализованный PnL увеличивает (или уменьшает) капитал и, значит, сайзинг следующих сделок.

События обрабатываются в едином хронологическом порядке (слияние баров всех инструментов),
поэтому одновременные входы конкурируют за один и тот же свободный остаток — как на реальном счёте.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

from bot_a.backtest.breakout_backtest import BreakoutResult, BreakoutTrade
from bot_a.backtest.candles import Candle
from bot_a.economics.fees import Tariff, TinkoffFeeCalculator
from bot_a.engine.breakout_fsm import (
    Bar,
    BreakoutEvent,
    BreakoutExitReason,
    BreakoutParams,
    BreakoutSide,
    MomentumBreakoutStateMachine,
)
from bot_a.models.domain import InstrumentMetadata
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import MSK

ZERO = Decimal(0)


@dataclass(slots=True)
class SwingAccountSettings:
    """Настройки общего счёта."""

    capital_rub: float = 50_000.0
    risk_pct_per_trade: float = 1.0
    tariff: Tariff = Tariff.PREMIUM
    margin_safety_buffer_pct: float = 25.0
    max_margin_utilization_pct: float = 60.0
    max_concurrent_positions: int = 0
    max_contracts_per_trade: int = 0


@dataclass(slots=True)
class AccountPosition:
    """Открытая позиция на общем счёте."""

    ticker: str
    side: BreakoutSide
    quantity: int
    margin_rub: float
    entry_price: Decimal
    entry_ms: int
    entry_fee: Decimal
    atr_points: float
    stop_points: float


@dataclass(slots=True)
class SwingAccountResult:
    """Итог портфельного прогона."""

    trades: List[BreakoutTrade] = field(default_factory=list)
    per_ticker: Dict[str, BreakoutResult] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    equity_curve: List[Tuple[int, float]] = field(default_factory=list)
    max_margin_used: float = 0.0
    rejected_by_margin: int = 0
    rejected_by_risk: int = 0
    rejected_by_slots: int = 0

    @property
    def net_rub(self) -> float:
        return sum(trade.net_rub for trade in self.trades)

    @property
    def fees_rub(self) -> float:
        return sum(trade.fees_rub for trade in self.trades)

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1][1] if self.equity_curve else 0.0


class SwingAccountBacktester:
    """Гоняет несколько инструментов через общий счёт с общей маржой."""

    def __init__(
        self,
        instruments: Sequence[Tuple[InstrumentMetadata, BreakoutParams]],
        settings: Optional[SwingAccountSettings] = None,
    ) -> None:
        self._instruments = list(instruments)
        self._settings = settings or SwingAccountSettings()
        self._fees = TinkoffFeeCalculator(self._settings.tariff)

    # ------------------------------------------------------------------ издержки
    def _notional(self, meta: InstrumentMetadata, price: Decimal, quantity: int) -> Decimal:
        return meta.value_in_rub(price) * Decimal(max(int(quantity), 1))

    def _round_trip_fee(self, meta: InstrumentMetadata, price: Decimal, quantity: int) -> Decimal:
        return self._fees.fee_for_notional(self._notional(meta, price, quantity)) * 2

    def _slip(self, meta: InstrumentMetadata, params: BreakoutParams) -> Decimal:
        return meta.min_price_increment * Decimal(max(int(params.slippage_steps), 0))

    # ------------------------------------------------------------------ сайзинг
    def risk_quantity(
        self, meta: InstrumentMetadata, stop_points: float, equity: Optional[float] = None
    ) -> int:
        """Сколько контрактов покрывает риск-бюджет (доля общего капитала)."""
        settings = self._settings
        if equity is None:
            equity = float(settings.capital_rub)
        risk_budget = max(equity, 0.0) * float(settings.risk_pct_per_trade) / 100.0
        stop_rub = max(float(stop_points) * float(meta.rub_per_price_unit), 1e-9)
        return int(risk_budget // stop_rub)

    def size(
        self,
        meta: InstrumentMetadata,
        stop_points: float,
        equity: Optional[float] = None,
        margin_used: float = 0.0,
    ) -> int:
        """Размер позиции от общего капитала и свободного ГО (0 = сделка отклонена)."""
        settings = self._settings
        if equity is None:
            equity = float(settings.capital_rub)
        by_risk = self.risk_quantity(meta, stop_points, equity)
        margin_per_contract = float(meta.position_margin_rub(1)) * (
            1.0 + float(settings.margin_safety_buffer_pct) / 100.0
        )
        # Ограничение утилизации не даёт занять весь счёт: иначе любое неблагоприятное
        # движение (убыток по открытым позициям) сразу вызовет маржин-колл.
        util_cap = max(equity, 0.0) * max(float(settings.max_margin_utilization_pct), 1.0) / 100.0
        free_cash = max(min(util_cap, equity) - margin_used, 0.0)
        by_margin = int(free_cash // margin_per_contract) if margin_per_contract > 0 else by_risk
        quantity = max(min(by_risk, by_margin), 0)
        if settings.max_contracts_per_trade > 0:
            quantity = min(quantity, settings.max_contracts_per_trade)
        return quantity

    # ------------------------------------------------------------------ прогон
    def run(self, data: Dict[str, Sequence[Candle]]) -> SwingAccountResult:
        settings = self._settings
        result = SwingAccountResult()
        machines: Dict[str, MomentumBreakoutStateMachine] = {}
        metas: Dict[str, InstrumentMetadata] = {}
        params_by_ticker: Dict[str, BreakoutParams] = {}
        open_positions: Dict[str, AccountPosition] = {}
        pending: Dict[str, Tuple[BreakoutEvent, AccountPosition]] = {}
        equity = float(settings.capital_rub)
        margin_used = 0.0

        for meta, params in self._instruments:
            ticker = meta.ticker
            metas[ticker] = meta
            params_by_ticker[ticker] = params
            machines[ticker] = MomentumBreakoutStateMachine(
                meta,
                params,
                fee_calculator=lambda price, qty, _meta=meta: self._round_trip_fee(_meta, price, qty),
            )
            result.per_ticker[ticker] = BreakoutResult(ticker=ticker)

        timeline: List[Tuple[int, int, str, Candle]] = []
        for index, (ticker, candles) in enumerate(sorted(data.items())):
            if ticker not in machines:
                continue
            timeline.extend((candle.time_ms, index, ticker, candle) for candle in candles)
        timeline.sort(key=lambda item: (item[0], item[1]))

        last_bar: Dict[str, Bar] = {}
        current_day: Optional[date] = None

        for time_ms, _, ticker, candle in timeline:
            bar = Bar(
                time_ms=candle.time_ms,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=int(candle.volume),
            )
            meta = metas[ticker]
            machine = machines[ticker]
            last_bar[ticker] = bar
            day = datetime.fromtimestamp(candle.time_ms / 1000, tz=MSK).date()
            if day != current_day:
                current_day = day
                self._fees.roll_day(day)

            for event in machine.on_bar(bar):
                if event.kind == "entry":
                    if (
                        settings.max_concurrent_positions > 0
                        and len(open_positions) >= settings.max_concurrent_positions
                    ):
                        machine.position = None
                        result.rejected_by_slots += 1
                        continue
                    position = self._size_entry(meta, machine, event, equity, margin_used, result)
                    if position is None:
                        continue
                    entry_fee = self._fees.register_notional(
                        self._notional(meta, event.price, position.quantity), meta.ticker
                    )
                    position.entry_fee = entry_fee
                    position.entry_price = event.price
                    position.entry_ms = event.time_ms
                    open_positions[ticker] = position
                    margin_used += position.margin_rub
                    result.max_margin_used = max(result.max_margin_used, margin_used)
                    pending[ticker] = (event, position)
                    continue
                if ticker not in pending:
                    machine.position = None
                    continue
                if event.kind == "partial":
                    entry_event, position = pending[ticker]
                    closed = max(min(int(event.quantity), position.quantity), 1)
                    share = position.margin_rub * closed / max(position.quantity, 1)
                    entry_fee_share = position.entry_fee * Decimal(closed) / Decimal(
                        max(position.quantity, 1)
                    )
                    exit_price = self._apply_exit_fill(
                        meta, params_by_ticker[ticker], event.price, event.side
                    )
                    exit_fee = self._fees.register_notional(
                        self._notional(meta, exit_price, closed), meta.ticker
                    )
                    trade = self._make_trade(
                        ticker,
                        meta,
                        entry_event,
                        event,
                        exit_price,
                        _slice_position(position, closed, entry_fee_share),
                        exit_fee,
                    )
                    result.trades.append(trade)
                    result.per_ticker[ticker].trades.append(trade)
                    equity += trade.net_rub
                    margin_used = max(margin_used - share, 0.0)
                    position.quantity -= closed
                    position.margin_rub -= share
                    position.entry_fee = position.entry_fee - entry_fee_share
                    result.equity_curve.append((event.time_ms, equity))
                    continue
                entry_event, position = pending.pop(ticker)
                exit_price = self._apply_exit_fill(meta, params_by_ticker[ticker], event.price, event.side)
                exit_fee = self._fees.register_notional(
                    self._notional(meta, exit_price, position.quantity), meta.ticker
                )
                trade = self._make_trade(
                    ticker,
                    meta,
                    entry_event,
                    event,
                    exit_price,
                    position,
                    exit_fee,
                )
                result.trades.append(trade)
                result.per_ticker[ticker].trades.append(trade)
                equity += trade.net_rub
                margin_used = max(margin_used - position.margin_rub, 0.0)
                open_positions.pop(ticker, None)
                result.equity_curve.append((event.time_ms, equity))

        # Позиции, оставшиеся к концу данных, закрываются по последнему закрытию.
        for ticker, (_entry_event, position) in list(pending.items()):
            machine = machines[ticker]
            bar = last_bar.get(ticker)
            if bar is None or machine.position is None:
                continue
            forced = machine.force_close(bar)
            if forced is None:
                continue
            meta = metas[ticker]
            exit_price = self._apply_exit_fill(meta, params_by_ticker[ticker], bar.close, forced.side)
            exit_fee = self._fees.register_notional(
                self._notional(meta, exit_price, position.quantity), meta.ticker
            )
            trade = self._make_trade(
                ticker, meta, None, forced, exit_price, position, exit_fee
            )
            result.trades.append(trade)
            result.per_ticker[ticker].trades.append(trade)
            equity += trade.net_rub
            result.equity_curve.append((bar.time_ms, equity))
            pending.pop(ticker, None)

        for ticker, machine in machines.items():
            block = result.per_ticker[ticker]
            block.counters = machine.counters.as_dict()
            bars = data.get(ticker) or ()
            block.bars = len(bars)
            if bars:
                block.first_ms = bars[0].time_ms
                block.last_ms = bars[-1].time_ms
            for key, value in block.counters.items():
                result.counters[key] = result.counters.get(key, 0) + value
        result.equity_curve.sort(key=lambda item: item[0])
        return result

    # ------------------------------------------------------------------ сайзинг
    def _size_entry(
        self,
        meta: InstrumentMetadata,
        machine: MomentumBreakoutStateMachine,
        event: BreakoutEvent,
        equity: float,
        margin_used: float,
        result: SwingAccountResult,
    ) -> Optional[AccountPosition]:
        settings = self._settings
        position = machine.position
        if position is None:
            return None
        params = machine._p  # noqa: SLF001 — нужен только slippage_steps
        stop_points = abs(float(position.entry_price - position.initial_stop))
        quantity = self.size(meta, stop_points, equity, margin_used)
        if quantity <= 0:
            machine.position = None
            if self.risk_quantity(meta, stop_points, equity) <= 0:
                result.rejected_by_risk += 1
            else:
                result.rejected_by_margin += 1
            return None
        margin_per_contract = float(meta.position_margin_rub(1)) * (
            1.0 + float(settings.margin_safety_buffer_pct) / 100.0
        )
        slip = self._slip(meta, params)
        fill = event.price + slip if event.side is BreakoutSide.LONG else event.price - slip
        position.quantity = quantity
        position.entry_price = fill
        position.stop_price = (
            fill - dec(stop_points) if event.side is BreakoutSide.LONG else fill + dec(stop_points)
        )
        position.initial_stop = position.stop_price
        position.fee_round_trip_rub = self._round_trip_fee(meta, fill, quantity)
        event.price = fill
        return AccountPosition(
            ticker=meta.ticker,
            side=event.side,
            quantity=quantity,
            margin_rub=margin_per_contract * quantity,
            entry_price=fill,
            entry_ms=event.time_ms,
            entry_fee=ZERO,
            atr_points=position.atr_points,
            stop_points=stop_points,
        )

    def _apply_exit_fill(
        self,
        meta: InstrumentMetadata,
        params: BreakoutParams,
        price: Decimal,
        side: BreakoutSide,
    ) -> Decimal:
        slip = self._slip(meta, params)
        return price - slip if side is BreakoutSide.LONG else price + slip

    # ------------------------------------------------------------------ сделка
    def _make_trade(
        self,
        ticker: str,
        meta: InstrumentMetadata,
        entry_event: Optional[BreakoutEvent],
        exit_event: BreakoutEvent,
        exit_price: Decimal,
        position: AccountPosition,
        exit_fee: Decimal,
    ) -> BreakoutTrade:
        entry_price = position.entry_price if entry_event is None else entry_event.price
        entry_ms = position.entry_ms if entry_event is None else entry_event.time_ms
        sign = Decimal(1) if exit_event.side is BreakoutSide.LONG else Decimal(-1)
        gross = float(
            (exit_price - entry_price)
            * sign
            * meta.rub_per_price_unit
            * Decimal(position.quantity)
        )
        fees = float(position.entry_fee + exit_fee)
        return BreakoutTrade(
            ticker=ticker,
            side=exit_event.side,
            quantity=position.quantity,
            entry_ms=entry_ms,
            exit_ms=exit_event.time_ms,
            entry_price=entry_price,
            exit_price=exit_price,
            gross_rub=gross,
            fees_rub=fees,
            net_rub=gross - fees,
            reason=exit_event.reason or BreakoutExitReason.DATA_END,
            atr_points=position.atr_points,
            mfe_points=_float_after(exit_event.note, "mfe="),
            hold_minutes=(exit_event.time_ms - entry_ms) / 60_000.0,
        )


def _float_after(note: str, key: str) -> float:
    for token in note.split():
        if token.startswith(key):
            try:
                return float(token[len(key) :])
            except ValueError:
                return 0.0
    return 0.0


def _slice_position(position: AccountPosition, quantity: int, entry_fee: Decimal) -> AccountPosition:
    """Копия позиции под частичное закрытие (``quantity`` контрактов)."""
    return AccountPosition(
        ticker=position.ticker,
        side=position.side,
        quantity=quantity,
        margin_rub=position.margin_rub * quantity / max(position.quantity, 1),
        entry_price=position.entry_price,
        entry_ms=position.entry_ms,
        entry_fee=entry_fee,
        atr_points=position.atr_points,
        stop_points=position.stop_points,
    )
