"""Учёт сделок и построение дневной статистики (PnL, просадка, Sharpe)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Sequence

from bot_a.models.domain import BacktestTotals, DailyBacktestSummary, TradeRecord
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import trading_day_key

ZERO = Decimal("0")


@dataclass(slots=True)
class DayBucket:
    """Накопитель статистики одного торгового дня."""

    day: str
    trades: int = 0
    wins: int = 0
    gross_rub: Decimal = ZERO
    fees_rub: Decimal = ZERO
    net_rub: Decimal = ZERO
    margin_sum_rub: Decimal = ZERO
    hold_seconds: float = 0.0
    cumulative_rub: Decimal = ZERO
    peak_rub: Decimal = ZERO
    max_drawdown_rub: Decimal = ZERO

    def add(self, trade: TradeRecord) -> None:
        self.trades += 1
        if trade.net_pnl_rub > 0:
            self.wins += 1
        self.gross_rub += trade.gross_pnl_rub
        self.fees_rub += trade.fees_rub
        self.net_rub += trade.net_pnl_rub
        self.margin_sum_rub += trade.margin_rub
        self.hold_seconds += trade.hold_seconds
        self.cumulative_rub += trade.net_pnl_rub
        if self.cumulative_rub > self.peak_rub:
            self.peak_rub = self.cumulative_rub
        drawdown = self.peak_rub - self.cumulative_rub
        if drawdown > self.max_drawdown_rub:
            self.max_drawdown_rub = drawdown

    @property
    def win_rate_pct(self) -> float:
        return (self.wins / self.trades * 100.0) if self.trades else 0.0

    @property
    def avg_hold_seconds(self) -> float:
        return (self.hold_seconds / self.trades) if self.trades else 0.0

    @property
    def net_pct_of_margin(self) -> float:
        if self.margin_sum_rub <= 0:
            return 0.0
        return float(self.net_rub / self.margin_sum_rub * Decimal(100))

    def to_summary(self) -> DailyBacktestSummary:
        return DailyBacktestSummary(
            date=self.day,
            trades=self.trades,
            wins=self.wins,
            win_rate_pct=self.win_rate_pct,
            gross_pnl_rub=self.gross_rub,
            fees_rub=self.fees_rub,
            net_pnl_rub=self.net_rub,
            net_pnl_pct_of_margin=self.net_pct_of_margin,
            max_drawdown_rub=self.max_drawdown_rub,
            avg_hold_seconds=self.avg_hold_seconds,
        )


class PortfolioLedger:
    """Журнал сделок с агрегацией по дням и итоговыми метриками."""

    def __init__(self, start_equity_rub: Decimal = ZERO) -> None:
        self._start_equity = dec(start_equity_rub)
        self._trades: List[TradeRecord] = []
        self._days: Dict[str, DayBucket] = {}

    @property
    def trades(self) -> Sequence[TradeRecord]:
        return tuple(self._trades)

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    @property
    def start_equity_rub(self) -> Decimal:
        return self._start_equity

    @property
    def net_pnl_rub(self) -> Decimal:
        return sum((trade.net_pnl_rub for trade in self._trades), ZERO)

    @property
    def fees_rub(self) -> Decimal:
        return sum((trade.fees_rub for trade in self._trades), ZERO)

    def register(self, trade: TradeRecord) -> DayBucket:
        key = trading_day_key(trade.closed_ms).isoformat()
        bucket = self._days.get(key)
        if bucket is None:
            bucket = DayBucket(day=key)
            self._days[key] = bucket
        bucket.add(trade)
        self._trades.append(trade)
        return bucket

    def daily_summaries(self) -> List[DailyBacktestSummary]:
        return [self._days[key].to_summary() for key in sorted(self._days.keys())]

    def daily_net_series(self) -> List[float]:
        return [float(self._days[key].net_rub) for key in sorted(self._days.keys())]

    def totals(self) -> BacktestTotals:
        summaries = self.daily_summaries()
        gross = sum((trade.gross_pnl_rub for trade in self._trades), ZERO)
        fees = sum((trade.fees_rub for trade in self._trades), ZERO)
        net = sum((trade.net_pnl_rub for trade in self._trades), ZERO)
        wins = [trade.net_pnl_rub for trade in self._trades if trade.net_pnl_rub > 0]
        losses = [trade.net_pnl_rub for trade in self._trades if trade.net_pnl_rub < 0]
        gross_profit = sum(wins, ZERO)
        gross_loss = abs(sum(losses, ZERO))
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        hold = sum((trade.hold_seconds for trade in self._trades), 0.0)
        avg_hold = hold / len(self._trades) if self._trades else 0.0
        win_rate = len(wins) / len(self._trades) * 100.0 if self._trades else 0.0
        fee_drag = float(fees / abs(gross) * Decimal(100)) if gross != 0 else 0.0
        profitable_days = sum(1 for summary in summaries if summary.net_pnl_rub > 0)
        return BacktestTotals(
            trades=len(self._trades),
            profitable_days=profitable_days,
            total_days=len(summaries),
            gross_pnl_rub=gross,
            fees_rub=fees,
            net_pnl_rub=net,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            sharpe_annual=_annualized_sharpe(self.daily_net_series()),
            avg_hold_seconds=avg_hold,
            fee_drag_pct=fee_drag,
            max_drawdown_rub=self._max_drawdown(),
        )

    def _max_drawdown(self) -> Decimal:
        equity = ZERO
        peak = ZERO
        worst = ZERO
        for trade in self._trades:
            equity += trade.net_pnl_rub
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > worst:
                worst = drawdown
        return worst

    def reset(self) -> None:
        self._trades.clear()
        self._days.clear()


def _annualized_sharpe(daily_net: Sequence[float]) -> float:
    """Годовой Sharpe по дневному ряду PnL (безрисковая ставка = 0, 252 дня)."""
    values = list(daily_net)
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    if std <= 1e-12:
        return 0.0
    return mean / std * math.sqrt(252.0)
