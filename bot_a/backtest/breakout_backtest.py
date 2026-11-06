"""Бэктестер стратегии v3.0 на минутных барах (FORTS).

Исполнение реалистичное:

* вход — по цене закрытия бара пробоя **плюс** `slippage_steps` шагов против сделки;
* выход по стопу/тейку — по цене стопа/тейка **плюс** проскальзывание;
* комиссия T-Invest `Premium` считается по номиналу контракта за каждую сторону
  через `TinkoffFeeCalculator` (дневные ступени тарифа учитываются по факту оборота);
* размер позиции — от капитала и ГО: `qty = min(риск/стоп_RUB, капитал/ГО)`.

Метрики: Net PnL, Profit Factor, Win Rate, Payoff (средний плюс / средний минус),
Max Drawdown в рублях и процентах, годовая доходность, число сделок и дней.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

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
class BreakoutSettings:
    """Настройки прогона: капитал, риск, тариф, ГО-буфер."""

    capital_rub: float = 50_000.0
    risk_pct_per_trade: float = 1.5
    tariff: Tariff = Tariff.PREMIUM
    max_contracts: int = 0
    margin_safety_buffer_pct: float = 25.0


@dataclass(slots=True)
class BreakoutTrade:
    """Одна завершённая сделка."""

    ticker: str
    side: BreakoutSide
    quantity: int
    entry_ms: int
    exit_ms: int
    entry_price: Decimal
    exit_price: Decimal
    gross_rub: float
    fees_rub: float
    net_rub: float
    reason: BreakoutExitReason
    atr_points: float
    mfe_points: float
    hold_minutes: float

    @property
    def entry_time(self) -> datetime:
        return datetime.fromtimestamp(self.entry_ms / 1000, tz=MSK)

    @property
    def exit_time(self) -> datetime:
        return datetime.fromtimestamp(self.exit_ms / 1000, tz=MSK)

    def to_dict(self) -> Dict[str, object]:
        return {
            "side": self.side.value,
            "quantity": self.quantity,
            "entry": self.entry_time.strftime("%Y-%m-%d %H:%M"),
            "exit": self.exit_time.strftime("%Y-%m-%d %H:%M"),
            "entryPrice": float(self.entry_price),
            "exitPrice": float(self.exit_price),
            "grossRub": round(self.gross_rub, 2),
            "feesRub": round(self.fees_rub, 2),
            "netRub": round(self.net_rub, 2),
            "reason": self.reason.value,
            "atrPoints": round(self.atr_points, 4),
            "mfePoints": round(self.mfe_points, 4),
            "holdMinutes": round(self.hold_minutes, 1),
        }


@dataclass(slots=True)
class BreakoutResult:
    """Результат прогона по одному инструменту."""

    ticker: str
    trades: List[BreakoutTrade] = field(default_factory=list)
    counters: Dict[str, int] = field(default_factory=dict)
    bars: int = 0
    first_ms: int = 0
    last_ms: int = 0

    # ------------------------------------------------------------------ метрики
    @property
    def net_rub(self) -> float:
        return sum(trade.net_rub for trade in self.trades)

    @property
    def gross_rub(self) -> float:
        return sum(trade.gross_rub for trade in self.trades)

    @property
    def fees_rub(self) -> float:
        return sum(trade.fees_rub for trade in self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for trade in self.trades if trade.net_rub > 0)

    @property
    def losses(self) -> int:
        return sum(1 for trade in self.trades if trade.net_rub <= 0)

    @property
    def win_rate(self) -> float:
        return self.wins / len(self.trades) if self.trades else 0.0

    @property
    def profit_factor(self) -> Optional[float]:
        profit = sum(trade.net_rub for trade in self.trades if trade.net_rub > 0)
        loss = -sum(trade.net_rub for trade in self.trades if trade.net_rub <= 0)
        if loss <= 1e-9:
            return None if profit <= 1e-9 else float("inf")
        return profit / loss

    @property
    def avg_winner(self) -> float:
        values = [trade.net_rub for trade in self.trades if trade.net_rub > 0]
        return statistics.fmean(values) if values else 0.0

    @property
    def avg_loser(self) -> float:
        values = [trade.net_rub for trade in self.trades if trade.net_rub <= 0]
        return statistics.fmean(values) if values else 0.0

    @property
    def payoff(self) -> Optional[float]:
        loser = abs(self.avg_loser)
        if loser <= 1e-9:
            return None if self.avg_winner <= 0 else float("inf")
        return self.avg_winner / loser

    @property
    def avg_hold_minutes(self) -> float:
        values = [trade.hold_minutes for trade in self.trades]
        return statistics.fmean(values) if values else 0.0

    @property
    def best_trade(self) -> float:
        return max((trade.net_rub for trade in self.trades), default=0.0)

    @property
    def worst_trade(self) -> float:
        return min((trade.net_rub for trade in self.trades), default=0.0)

    @property
    def max_drawdown_rub(self) -> float:
        peak = 0.0
        worst = 0.0
        total = 0.0
        for trade in sorted(self.trades, key=lambda item: item.exit_ms):
            total += trade.net_rub
            peak = max(peak, total)
            worst = max(worst, peak - total)
        return worst

    def max_drawdown_pct(self, capital: float) -> float:
        return self.max_drawdown_rub / capital * 100.0 if capital > 0 else 0.0

    @property
    def days(self) -> int:
        return len(
            {datetime.fromtimestamp(trade.entry_ms / 1000, tz=MSK).date() for trade in self.trades}
        )

    def daily_pnl(self) -> Dict[date, float]:
        """Реализованный PnL по дням закрытия сделок (MSK)."""
        result: Dict[date, float] = {}
        for trade in self.trades:
            day = datetime.fromtimestamp(trade.exit_ms / 1000, tz=MSK).date()
            result[day] = result.get(day, 0.0) + trade.net_rub
        return result

    def sharpe(self, capital: float, trading_days_per_year: int = 252) -> Optional[float]:
        """Годовой Sharpe по дневным доходностям капитала (0 = нет сделок в этот день)."""
        if capital <= 0 or not self.trades:
            return None
        curve = self.daily_pnl()
        if len(curve) < 3:
            return None
        returns = [value / capital for value in curve.values()]
        mean = statistics.fmean(returns)
        sigma = statistics.pstdev(returns) if len(returns) > 1 else 0.0
        if sigma <= 1e-12:
            return None
        # Дни без сделок дают нулевую доходность — они тоже часть ряда (честнее для Sharpe).
        return mean / sigma * (trading_days_per_year ** 0.5)

    def annualized_pct(self, capital: float) -> float:
        """Годовая доходность по сложному проценту за фактический период данных."""
        if capital <= 0 or self.first_ms <= 0 or self.last_ms <= self.first_ms:
            return 0.0
        years = (self.last_ms - self.first_ms) / 1000.0 / (365.25 * 24 * 3600)
        if years <= 0:
            return 0.0
        growth = 1.0 + self.net_rub / capital
        if growth <= 0:
            return -100.0
        return (growth ** (1.0 / years) - 1.0) * 100.0

    def summary(self, capital: float) -> Dict[str, object]:
        return {
            "ticker": self.ticker,
            "bars": self.bars,
            "trades": len(self.trades),
            "days": self.days,
            "netRub": round(self.net_rub, 2),
            "grossRub": round(self.gross_rub, 2),
            "feesRub": round(self.fees_rub, 2),
            "winRate": round(self.win_rate, 4),
            "profitFactor": None if self.profit_factor is None else round(self.profit_factor, 3),
            "avgWinner": round(self.avg_winner, 2),
            "avgLoser": round(self.avg_loser, 2),
            "payoff": None if self.payoff is None else round(self.payoff, 3),
            "maxDrawdownRub": round(self.max_drawdown_rub, 2),
            "maxDrawdownPct": round(self.max_drawdown_pct(capital), 2),
            "returnPct": round(self.net_rub / capital * 100.0, 2) if capital else 0.0,
            "annualizedPct": round(self.annualized_pct(capital), 2),
            "avgHoldMinutes": round(self.avg_hold_minutes, 1),
            "bestTrade": round(self.best_trade, 2),
            "worstTrade": round(self.worst_trade, 2),
            "sharpe": None if self.sharpe(capital) is None else round(self.sharpe(capital), 2),
            "counters": dict(self.counters),
        }


def candles_to_bars(candles: Sequence[Candle]) -> List[Bar]:
    return [
        Bar(
            time_ms=candle.time_ms,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=int(candle.volume),
            adj=float(getattr(candle, "adj", 0.0) or 0.0),
        )
        for candle in candles
    ]


class BreakoutBacktester:
    """Прогоняет бары через `MomentumBreakoutStateMachine`, применяя издержки и ГО."""

    def __init__(
        self,
        meta: InstrumentMetadata,
        params: Optional[BreakoutParams] = None,
        settings: Optional[BreakoutSettings] = None,
    ) -> None:
        self._meta = meta
        self._p = params or BreakoutParams()
        self._settings = settings or BreakoutSettings()
        self._fees = TinkoffFeeCalculator(self._settings.tariff)
        self._slippage = meta.min_price_increment * Decimal(max(int(self._p.slippage_steps), 0))

    @property
    def settings(self) -> BreakoutSettings:
        return self._settings

    # ------------------------------------------------------------------ размер позиции
    def size(self, stop_points: float) -> int:
        """Контракты: лимит по риску на сделку и по ГО с буфером."""
        settings = self._settings
        capital = max(float(settings.capital_rub), 0.0)
        rub_per_point = float(self._meta.rub_per_price_unit)
        stop_rub_per_contract = max(stop_points * rub_per_point, 1e-9)
        risk_rub = capital * float(settings.risk_pct_per_trade) / 100.0
        by_risk = int(risk_rub // stop_rub_per_contract) if risk_rub > 0 else 0
        margin = float(self._meta.position_margin_rub(1)) * (
            1.0 + float(settings.margin_safety_buffer_pct) / 100.0
        )
        by_margin = int(capital // margin) if margin > 0 else by_risk
        quantity = max(min(by_risk, by_margin), 0)
        if settings.max_contracts > 0:
            quantity = min(quantity, settings.max_contracts)
        return quantity

    # ------------------------------------------------------------------ издержки
    def _notional(self, price: Decimal, quantity: int) -> Decimal:
        return self._meta.value_in_rub(price) * Decimal(max(int(quantity), 1))

    def _round_trip_fee(self, price: Decimal, quantity: int) -> Decimal:
        """Оценка круговой комиссии без изменения состояния (фильтр ATR/fee)."""
        return self._fees.fee_for_notional(self._notional(price, quantity)) * 2

    # ------------------------------------------------------------------ прогон
    def run(self, candles: Sequence[Candle]) -> BreakoutResult:
        bars = candles_to_bars(candles)
        machine = regime_machine(
            self._meta,
            self._p,
            fee_calculator=self._round_trip_fee,
        )
        result = BreakoutResult(ticker=self._meta.ticker, bars=len(bars))
        if bars:
            result.first_ms = bars[0].time_ms
            result.last_ms = bars[-1].time_ms
        current_day: Optional[date] = None
        pending: Optional[Tuple[BreakoutEvent, int, Decimal, float, int]] = None
        no_capital = 0
        for bar in bars:
            day = datetime.fromtimestamp(bar.time_ms / 1000, tz=MSK).date()
            if day != current_day:
                current_day = day
                self._fees.roll_day(day)
            for event in machine.on_bar(bar):
                if event.kind == "entry":
                    pending = self._fill_entry(machine, event)
                    if pending is None:
                        no_capital += 1
                    continue
                if pending is None:
                    continue
                if event.kind == "partial":
                    result.trades.append(self._close_partial(pending, event))
                    pending = self._reduce_pending(pending, event.quantity)
                    continue
                result.trades.append(self._close_pending(pending, event))
                pending = None
        if pending is not None and machine.position is not None and bars:
            forced = machine.force_close(bars[-1])
            if forced is not None:
                result.trades.append(self._close_pending(pending, forced))
        counters = machine.counters.as_dict()
        if no_capital:
            counters["noCapitalSkips"] = no_capital
        result.counters = counters
        return result

    # ------------------------------------------------------------------ исполнение
    def _fill_entry(
        self, machine: MomentumBreakoutStateMachine, event: BreakoutEvent
    ) -> Optional[Tuple[BreakoutEvent, int, Decimal, float, int]]:
        position = machine.position
        if position is None:
            return None
        stop_points = abs(float(position.entry_price - position.initial_stop))
        quantity = self.size(stop_points)
        if quantity <= 0:
            machine.position = None
            return None
        fill, stop = self._apply_entry_fill(event.price, event.side, stop_points)
        position.quantity = quantity
        position.entry_price = fill
        position.stop_price = stop
        position.initial_stop = stop
        position.fee_round_trip_rub = self._round_trip_fee(fill, quantity)
        if event.side is BreakoutSide.LONG:
            position.max_price = max(position.max_price, fill)
        else:
            position.min_price = fill if position.min_price <= 0 else min(position.min_price, fill)
        event.price = fill
        entry_fee = self._fees.register_notional(self._notional(fill, quantity))
        return (event, quantity, entry_fee, position.atr_points, position.entry_ms)

    def _close_pending(
        self, pending: Tuple[BreakoutEvent, int, Decimal, float, int], event: BreakoutEvent
    ) -> BreakoutTrade:
        entry_event, quantity, entry_fee, atr_points, entry_ms = pending
        exit_price = self._apply_exit_fill(event.price, event.side)
        exit_fee = self._fees.register_notional(self._notional(exit_price, quantity))
        sign = Decimal(1) if event.side is BreakoutSide.LONG else Decimal(-1)
        gross = float(
            (exit_price - entry_event.price) * sign * self._meta.rub_per_price_unit * Decimal(quantity)
        )
        fees = float(entry_fee + exit_fee)
        return BreakoutTrade(
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
            mfe_points=_float_after(event.note, "mfe="),
            hold_minutes=(event.time_ms - entry_ms) / 60_000.0,
        )

    def _apply_entry_fill(
        self, price: Decimal, side: BreakoutSide, stop_points: float
    ) -> Tuple[Decimal, Decimal]:
        distance = dec(stop_points)
        if side is BreakoutSide.LONG:
            fill = price + self._slippage
            return fill, fill - distance
        fill = price - self._slippage
        return fill, fill + distance

    def _close_partial(
        self, pending: Tuple[BreakoutEvent, int, Decimal, float, int], event: BreakoutEvent
    ) -> BreakoutTrade:
        """Частичная фиксация: закрываем ``event.quantity`` контрактов из позиции."""
        entry_event, quantity, entry_fee, atr_points, entry_ms = pending
        closed = max(min(int(event.quantity), quantity), 1)
        share = entry_fee * Decimal(closed) / Decimal(quantity)
        exit_price = self._apply_exit_fill(event.price, event.side)
        exit_fee = self._fees.register_notional(self._notional(exit_price, closed))
        sign = Decimal(1) if event.side is BreakoutSide.LONG else Decimal(-1)
        gross = float(
            (exit_price - entry_event.price) * sign * self._meta.rub_per_price_unit * Decimal(closed)
        )
        fees = float(share + exit_fee)
        return BreakoutTrade(
            ticker=self._meta.ticker,
            side=event.side,
            quantity=closed,
            entry_ms=entry_event.time_ms,
            exit_ms=event.time_ms,
            entry_price=entry_event.price,
            exit_price=exit_price,
            gross_rub=gross,
            fees_rub=fees,
            net_rub=gross - fees,
            reason=event.reason or BreakoutExitReason.PARTIAL,
            atr_points=atr_points,
            mfe_points=_float_after(event.note, "mfe="),
            hold_minutes=(event.time_ms - entry_ms) / 60_000.0,
        )

    @staticmethod
    def _reduce_pending(
        pending: Tuple[BreakoutEvent, int, Decimal, float, int], closed: int
    ) -> Tuple[BreakoutEvent, int, Decimal, float, int]:
        entry_event, quantity, entry_fee, atr_points, entry_ms = pending
        remaining = max(quantity - max(int(closed), 1), 0)
        share = entry_fee * Decimal(remaining) / Decimal(quantity) if quantity else ZERO
        return (entry_event, remaining, share, atr_points, entry_ms)

    def _apply_exit_fill(self, price: Decimal, side: BreakoutSide) -> Decimal:
        return price - self._slippage if side is BreakoutSide.LONG else price + self._slippage


def _float_after(note: str, key: str) -> float:
    for token in note.split():
        if token.startswith(key):
            try:
                return float(token[len(key) :])
            except ValueError:
                return 0.0
    return 0.0


def aggregate(results: Sequence[BreakoutResult], capital: float) -> Dict[str, object]:
    """Свод по портфелю (деньги складываются, доля — от общего капитала)."""
    trades = [trade for result in results for trade in result.trades]
    combined = BreakoutResult(
        ticker="PORTFOLIO",
        trades=trades,
        bars=sum(result.bars for result in results),
        first_ms=min((result.first_ms for result in results if result.first_ms), default=0),
        last_ms=max((result.last_ms for result in results if result.last_ms), default=0),
    )
    summary = combined.summary(capital)
    summary["tickers"] = [result.ticker for result in results]
    return summary


def portfolio_curve(
    results: Sequence[BreakoutResult], capital: float
) -> Dict[str, object]:
    """Объединённая кривая эквити портфеля: сделки всех инструментов по времени выхода.

    Возвращает точки ``[(дата, накопленный PnL)]`` (по дню закрытия) и статистику:
    Net, Max DD в рублях и процентах капитала, Sharpe по дневным доходностям.
    """
    daily: Dict[date, float] = {}
    for result in results:
        for day, value in result.daily_pnl().items():
            daily[day] = daily.get(day, 0.0) + value
    if not daily:
        return {
            "points": [],
            "netRub": 0.0,
            "maxDrawdownRub": 0.0,
            "maxDrawdownPct": 0.0,
            "sharpe": None,
            "days": 0,
            "bestDay": 0.0,
            "worstDay": 0.0,
        }
    points: List[Tuple[str, float]] = []
    total = 0.0
    peak = 0.0
    worst = 0.0
    for day in sorted(daily):
        total += daily[day]
        peak = max(peak, total)
        worst = max(worst, peak - total)
        points.append((day.isoformat(), round(total, 2)))
    values = [daily[day] / capital for day in sorted(daily)] if capital > 0 else []
    sharpe = None
    if len(values) >= 3:
        mean = statistics.fmean(values)
        sigma = statistics.pstdev(values)
        if sigma > 1e-12:
            sharpe = mean / sigma * (252 ** 0.5)
    monthly: Dict[str, float] = {}
    for day, value in daily.items():
        key = f"{day.year}-{day.month:02d}"
        monthly[key] = monthly.get(key, 0.0) + value
    return {
        "points": points,
        "monthly": {key: round(value, 2) for key, value in sorted(monthly.items())},
        "netRub": round(sum(daily.values()), 2),
        "maxDrawdownRub": round(worst, 2),
        "maxDrawdownPct": round(worst / capital * 100.0, 2) if capital > 0 else 0.0,
        "sharpe": None if sharpe is None else round(sharpe, 2),
        "days": len(daily),
        "bestDay": round(max(daily.values()), 2),
        "worstDay": round(min(daily.values()), 2),
    }
