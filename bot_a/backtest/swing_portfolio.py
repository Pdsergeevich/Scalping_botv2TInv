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
from bot_a.engine.multiregime_fsm import regime_machine
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
    # v4.1: сторож здоровья инструмента (rolling health guard). Всё считается
    # только по уже закрытым сделкам — заглядывания вперёд нет.
    health_guard: bool = False
    rolling_window_trades: int = 8
    rolling_pf_floor: float = 0.70
    consecutive_loss_limit: int = 3
    # v4.1: торможение риска на просадке (только по уже реализованному PnL).
    drawdown_throttle: bool = False
    throttle_start_pct: float = 8.0
    throttle_min_multiplier: float = 0.4
    # v4.2: сайзинг по уверенности (сила сигнала × ранг инструмента).
    conviction_sizing: bool = False
    conviction_min: float = 0.75
    conviction_max: float = 1.35


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
    virtual: bool = False  # «виртуальная» сделка при паузе: считается, но не торгуется


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
    health_pauses: int = 0
    health_resumes: int = 0

    def metrics(self, capital: float) -> Dict[str, Any]:
        """Метрики ТОЛЬКО по фактической кривой equity общего счёта (v4.2)."""
        return equity_metrics(self.equity_curve, self.trades, capital)

    @property
    def net_rub(self) -> float:
        return sum(trade.net_rub for trade in self.trades)

    @property
    def fees_rub(self) -> float:
        return sum(trade.fees_rub for trade in self.trades)

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1][1] if self.equity_curve else 0.0


def equity_metrics(
    equity_curve: Sequence[Tuple[int, float]],
    trades: Sequence[BreakoutTrade],
    capital: float,
) -> Dict[str, Any]:
    """PF, Sharpe, Max DD (₽ и %), месяцы — по фактической кривой общего счёта (v4.2).

    Раньше метрики считались по `portfolio_curve` (сумма изолированных кривых),
    из-за чего просадка общего счёта отличалась от брокерской. Теперь всё берётся
    из реальной equity-траектории: пик → просадка, дневные доходности → Sharpe.
    """
    points = [(int(ms), float(value)) for ms, value in equity_curve]
    if not points:
        return {
            "netRub": 0.0, "profitFactor": None, "sharpe": None, "maxDrawdownRub": 0.0,
            "maxDrawdownPct": 0.0, "returnPct": 0.0, "annualizedPct": 0.0, "monthly": {},
            "trades": len(trades), "days": 0, "winRate": 0.0,
        }
    net = points[-1][1] - float(capital)
    profits = sum(trade.net_rub for trade in trades if trade.net_rub > 0)
    losses = -sum(trade.net_rub for trade in trades if trade.net_rub <= 0)
    peak = float(capital)
    max_dd_rub = 0.0
    max_dd_pct = 0.0
    for _, value in points:
        peak = max(peak, value)
        drawdown = peak - value
        if drawdown > max_dd_rub:
            max_dd_rub = drawdown
            max_dd_pct = drawdown / peak * 100.0 if peak > 0 else 0.0
    daily: Dict[str, float] = {}
    for time_ms, value in points:
        daily[datetime.fromtimestamp(time_ms / 1000, tz=MSK).date().isoformat()] = value
    series = [daily[key] for key in sorted(daily)]
    returns = [
        series[index] / series[index - 1] - 1.0
        for index in range(1, len(series))
        if series[index - 1] > 0
    ]
    sharpe = None
    if len(returns) >= 5:
        mean = sum(returns) / len(returns)
        variance = sum((item - mean) ** 2 for item in returns) / max(len(returns) - 1, 1)
        deviation = variance ** 0.5
        if deviation > 0:
            sharpe = mean / deviation * (252 ** 0.5)
    monthly: Dict[str, float] = {}
    previous = float(capital)
    for key in sorted(daily):
        value = daily[key]
        monthly[key[:7]] = monthly.get(key[:7], 0.0) + (value - previous)
        previous = value
    keys = sorted(daily)
    span_days = (
        date.fromisoformat(keys[-1]) - date.fromisoformat(keys[0])
    ).days + 1 if len(keys) > 1 else 1
    winners = sum(1 for trade in trades if trade.net_rub > 0)
    return {
        "netRub": round(net, 2),
        "profitFactor": round(profits / losses, 3) if losses > 0 else None,
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "maxDrawdownRub": round(max_dd_rub, 2),
        "maxDrawdownPct": round(max_dd_pct, 2),
        "returnPct": round(net / float(capital) * 100.0, 2) if capital else 0.0,
        "annualizedPct": round(net / float(capital) * 365.0 / max(span_days, 1) * 100.0, 2)
        if capital
        else 0.0,
        "monthly": {key: round(value, 2) for key, value in monthly.items()},
        "trades": len(trades),
        "days": len(daily),
        "winRate": round(winners / len(trades), 4) if trades else 0.0,
    }


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
        self._risk_multiplier = 1.0
        # v4.2: кратность риска по уверенности (только по закрытым сделкам).
        self._conviction = 1.0
        self._ticker_pf: Dict[str, float] = {}

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
        conviction = self._conviction_multiplier(meta.ticker)
        by_risk = int(
            self.risk_quantity(meta, stop_points, equity) * self._risk_multiplier * conviction
        )
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
        self._result_ref = result

        for meta, params in self._instruments:
            ticker = meta.ticker
            metas[ticker] = meta
            params_by_ticker[ticker] = params
            machines[ticker] = regime_machine(
                meta,
                params,
                fee_calculator=lambda price, qty, _meta=meta: self._round_trip_fee(_meta, price, qty),
            )
            result.per_ticker[ticker] = BreakoutResult(ticker=ticker)

        # v4.1: история последних сделок по инструменту для сторожа здоровья.
        health: Dict[str, List[float]] = {ticker: [] for ticker in machines}
        paused: Dict[str, bool] = {ticker: False for ticker in machines}
        losing_streak: Dict[str, int] = {ticker: 0 for ticker in machines}
        peak_equity = float(settings.capital_rub)

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
                    if paused.get(ticker):
                        # Сторож на паузе: сделку не исполняем, но ведём «виртуально»,
                        # чтобы увидеть первый виртуальный плюс и вернуть инструмент.
                        virtual = self._open_virtual(meta, machine, event)
                        if virtual is not None:
                            open_positions[ticker] = virtual
                            pending[ticker] = (event, virtual)
                        continue
                    if (
                        settings.max_concurrent_positions > 0
                        and len(open_positions) >= settings.max_concurrent_positions
                    ):
                        machine.position = None
                        result.rejected_by_slots += 1
                        continue
                    self._conviction = float(getattr(event, "conviction", 1.0) or 1.0)
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
                    peak_equity = max(peak_equity, equity)
                    self._risk_multiplier = self._throttle_multiplier(equity, peak_equity)
                    margin_used = max(margin_used - share, 0.0)
                    position.quantity -= closed
                    position.margin_rub -= share
                    position.entry_fee = position.entry_fee - entry_fee_share
                    result.equity_curve.append((event.time_ms, equity))
                    continue
                entry_event, position = pending.pop(ticker)
                if position.virtual:
                    self._settle_virtual(
                        ticker, meta, params_by_ticker[ticker], entry_event, position, event,
                        settings, health, paused, losing_streak, result,
                    )
                    open_positions.pop(ticker, None)
                    continue
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
                peak_equity = max(peak_equity, equity)
                self._risk_multiplier = self._throttle_multiplier(equity, peak_equity)
                margin_used = max(margin_used - position.margin_rub, 0.0)
                open_positions.pop(ticker, None)
                result.equity_curve.append((event.time_ms, equity))
                if settings.health_guard:
                    self._track_health(ticker, trade.net_rub, settings, health, paused, losing_streak, result)
                if settings.conviction_sizing:
                    self._update_ticker_pf(ticker)

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

    def _conviction_multiplier(self, ticker: str) -> float:
        """Множитель риска: сила сигнала × ранг инструмента, в [conviction_min, conviction_max]."""
        settings = self._settings
        if not settings.conviction_sizing:
            return 1.0
        multiplier = self._conviction
        trailing_pf = self._ticker_pf.get(ticker)
        if trailing_pf is not None:
            if trailing_pf >= 1.80:
                multiplier *= 1.20
            elif trailing_pf < 1.10:
                multiplier *= 0.80
        low = max(float(settings.conviction_min), 0.1)
        high = max(float(settings.conviction_max), low)
        return min(max(multiplier, low), high)

    def _update_ticker_pf(self, ticker: str, window: int = 8) -> None:
        """Скользящий PF инструмента по последним закрытым сделкам (без lookahead)."""
        block = self._result_ref.per_ticker.get(ticker) if self._result_ref else None
        if block is None:
            return
        recent = block.trades[-window:]
        if len(recent) < 4:
            return
        profits = sum(trade.net_rub for trade in recent if trade.net_rub > 0)
        losses = -sum(trade.net_rub for trade in recent if trade.net_rub <= 0)
        if losses <= 0:
            return
        self._ticker_pf[ticker] = profits / losses

    def _throttle_multiplier(self, equity: float, peak_equity: float) -> float:
        """Линейное торможение риска на просадке: 1.0 → throttle_min_multiplier.

        Считается только по реализованному PnL, поэтому заглядывания вперёд нет.
        """
        settings = self._settings
        if not settings.drawdown_throttle or peak_equity <= 0:
            return 1.0
        drawdown = max(peak_equity - equity, 0.0) / peak_equity * 100.0
        start = max(float(settings.throttle_start_pct), 0.1)
        if drawdown <= start:
            return 1.0
        span = max(30.0 - start, 1.0)
        scale = min((drawdown - start) / span, 1.0)
        floor = max(min(float(settings.throttle_min_multiplier), 1.0), 0.1)
        return 1.0 - (1.0 - floor) * scale

    # ------------------------------------------------------------------ сторож здоровья
    def _open_virtual(
        self,
        meta: InstrumentMetadata,
        machine: MomentumBreakoutStateMachine,
        event: BreakoutEvent,
    ) -> Optional[AccountPosition]:
        """Виртуальная позиция на паузе: FSM работает, счёт не трогаем."""
        position = machine.position
        if position is None:
            return None
        position.quantity = 1
        position.fee_round_trip_rub = self._round_trip_fee(meta, event.price, 1)
        return AccountPosition(
            ticker=meta.ticker,
            side=event.side,
            quantity=1,
            margin_rub=0.0,
            entry_price=event.price,
            entry_ms=event.time_ms,
            entry_fee=ZERO,
            atr_points=position.atr_points,
            stop_points=abs(float(position.entry_price - position.initial_stop)),
            virtual=True,
        )

    def _settle_virtual(
        self,
        ticker: str,
        meta: InstrumentMetadata,
        params: BreakoutParams,
        entry_event: BreakoutEvent,
        position: AccountPosition,
        event: BreakoutEvent,
        settings: SwingAccountSettings,
        health: Dict[str, List[float]],
        paused: Dict[str, bool],
        losing_streak: Dict[str, int],
        result: SwingAccountResult,
    ) -> None:
        """Виртуальный выход: считаем гипотетический PnL, но ничего не списываем."""
        exit_price = self._apply_exit_fill(meta, params, event.price, event.side)
        sign = 1 if event.side is BreakoutSide.LONG else -1
        gross = float((exit_price - entry_event.price) * sign * meta.rub_per_price_unit)
        fee = float(self._round_trip_fee(meta, exit_price, 1))
        net = gross - fee
        if net > 0:
            paused[ticker] = False
            losing_streak[ticker] = 0
            health[ticker].append(net)
            result.health_resumes += 1
        else:
            health[ticker].append(net)

    def _track_health(
        self,
        ticker: str,
        net: float,
        settings: SwingAccountSettings,
        health: Dict[str, List[float]],
        paused: Dict[str, bool],
        losing_streak: Dict[str, int],
        result: SwingAccountResult,
    ) -> None:
        """Пауза инструмента при 3 убытках подряд или PF < floor на скользящем окне."""
        history = health[ticker]
        history.append(net)
        if len(history) > max(int(settings.rolling_window_trades), 2):
            history.pop(0)
        losing_streak[ticker] = losing_streak.get(ticker, 0) + 1 if net <= 0 else 0
        if paused[ticker]:
            return
        window = history[-max(int(settings.rolling_window_trades), 2) :]
        profits = sum(item for item in window if item > 0)
        losses = -sum(item for item in window if item <= 0)
        bad_pf = len(window) >= max(int(settings.rolling_window_trades), 2) and (
            losses > 0 and profits / losses < float(settings.rolling_pf_floor)
        )
        if losing_streak[ticker] >= max(int(settings.consecutive_loss_limit), 1) or bad_pf:
            paused[ticker] = True
            result.health_pauses += 1

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
