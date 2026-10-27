"""Доменные модели Bot A.

Все рыночные структуры объявлены как ``frozen`` + ``slots`` датаклассы —
аналог ``readonly record struct`` из C#: неизменяемы, без словаря атрибутов,
дешёвы в создании на «горячем» пути обработки тиков.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional, Tuple

from bot_a.utils.quotation import (
    ceil_to_increment,
    floor_to_increment,
    round_to_increment,
    steps_between,
)

ZERO = Decimal("0")


class Side(str, Enum):
    """Направление сделки."""

    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY

    @staticmethod
    def from_sign(value: int) -> "Side":
        return Side.BUY if value > 0 else Side.SELL


class MarketKind(str, Enum):
    """Тип инструмента: спот (TQBR) или срочный рынок (SPBFUT)."""

    SPOT = "SPOT"
    FUTURES = "FUTURES"


class Liquidity(str, Enum):
    """Тип ликвидности исполнения: мейкер (пассив) или тейкер (агрессия)."""

    MAKER = "maker"
    TAKER = "taker"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"
    LIMIT_IOC = "limit_ioc"


class OrderStatus(str, Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)


class ExecutionTopology(str, Enum):
    """Топология исполнения пары."""

    DUAL_LEG_HEDGED = "DualLegHedged"
    SINGLE_LEG_LEAD_LAG = "SingleLegLeadLag"


class FsmState(str, Enum):
    """Состояния конечного автомата исполнения."""

    WARMUP = "Warmup"
    IDLE = "Idle"
    QUOTING_ENTRY_LEG1 = "QuotingEntryLeg1"
    HEDGING_ENTRY_LEG2 = "HedgingEntryLeg2"
    IN_POSITION = "InPosition"
    QUOTING_EXIT_LEG1 = "QuotingExitLeg1"
    HEDGING_EXIT_LEG2 = "HedgingExitLeg2"
    EMERGENCY_FLATTEN = "EmergencyFlatten"
    COOLDOWN = "Cooldown"


class ExitReason(str, Enum):
    TAKE_PROFIT = "TakeProfit"
    TIME_STOP = "TimeStop"
    HARD_STOP = "HardStop"
    EMERGENCY = "EmergencyFlatten"
    SHUTDOWN = "Shutdown"
    SCRATCH = "Scratch"
    BREAKEVEN = "Breakeven"


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    price: Decimal
    quantity: int


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """Снимок стакана (top-N) на момент времени ``time_ms`` (UTC, миллисекунды)."""

    instrument_uid: str
    time_ms: int
    bids: Tuple[OrderBookLevel, ...]
    asks: Tuple[OrderBookLevel, ...]
    last_price: Optional[Decimal] = None

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0].price if self.asks else None

    @property
    def has_two_sided_book(self) -> bool:
        return bool(self.bids) and bool(self.asks)

    @property
    def bid_volume(self) -> int:
        return sum(level.quantity for level in self.bids)

    @property
    def ask_volume(self) -> int:
        return sum(level.quantity for level in self.asks)


@dataclass(frozen=True, slots=True)
class TradePrint:
    """Реальный принт сделки (лента обезличенных сделок)."""

    instrument_uid: str
    time_ms: int
    price: Decimal
    quantity: int
    side: Optional[Side] = None
    trade_id: str = ""


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """Унифицированное событие рынка: либо стакан, либо принт сделки."""

    time_ms: int
    leg: int
    book: Optional[OrderBookSnapshot] = None
    trade: Optional[TradePrint] = None

    @property
    def is_book(self) -> bool:
        return self.book is not None

    @staticmethod
    def of_book(leg: int, book: OrderBookSnapshot) -> "MarketEvent":
        return MarketEvent(time_ms=book.time_ms, leg=leg, book=book)

    @staticmethod
    def of_trade(leg: int, trade: TradePrint) -> "MarketEvent":
        return MarketEvent(time_ms=trade.time_ms, leg=leg, trade=trade)


@dataclass(frozen=True, slots=True)
class InstrumentMetadata:
    """Метаданные инструмента и экономика пункта цены.

    Для фьючерсов (SPBFUT) цена котируется в пунктах, а стоимость позиции и PnL
    считаются в рублях через ``MinPriceIncrement`` / ``MinPriceIncrementAmount``.
    Для акций (TQBR) рублёвая стоимость единицы цены равна размеру лота.
    """

    instrument_uid: str
    ticker: str
    class_code: str
    market: MarketKind = MarketKind.SPOT
    figi: str = ""
    name: str = ""
    lot_size: int = 1
    min_price_increment: Decimal = Decimal("0.01")
    min_price_increment_amount: Decimal = Decimal("0")
    initial_margin_buy: Decimal = Decimal("0")
    initial_margin_sell: Decimal = Decimal("0")
    currency: str = "rub"

    @property
    def is_futures(self) -> bool:
        return self.market is MarketKind.FUTURES

    @property
    def rub_per_price_unit(self) -> Decimal:
        """Рублёвая стоимость одной единицы цены (1 пункт для фьючерса)."""
        if self.is_futures and self.min_price_increment > 0:
            amount = self.min_price_increment_amount
            if amount <= 0:
                amount = self.min_price_increment
            return amount / self.min_price_increment
        return Decimal(self.lot_size if self.lot_size > 0 else 1)

    def value_in_rub(self, price: Decimal) -> Decimal:
        """Стоимость контракта/лота в рублях по цене ``price``."""
        return Decimal(price) * self.rub_per_price_unit

    def position_margin_rub(self, lots: int) -> Decimal:
        """Требуемое ГО (руб.) на ``lots`` контрактов."""
        per_contract = max(self.initial_margin_buy, self.initial_margin_sell)
        return per_contract * Decimal(abs(lots))

    def round_price(self, price: Decimal) -> Decimal:
        return round_to_increment(price, self.min_price_increment)

    def floor_price(self, price: Decimal) -> Decimal:
        return floor_to_increment(price, self.min_price_increment)

    def ceil_price(self, price: Decimal) -> Decimal:
        return ceil_to_increment(price, self.min_price_increment)

    def price_steps(self, price_from: Decimal, price_to: Decimal) -> int:
        return steps_between(price_from, price_to, self.min_price_increment)

    def with_uid(self, instrument_uid: str) -> "InstrumentMetadata":
        from dataclasses import replace

        return replace(self, instrument_uid=instrument_uid)

    def to_dict(self) -> dict:
        return {
            "instrumentUid": self.instrument_uid,
            "ticker": self.ticker,
            "classCode": self.class_code,
            "market": self.market.value,
            "figi": self.figi,
            "name": self.name,
            "lotSize": self.lot_size,
            "minPriceIncrement": str(self.min_price_increment),
            "minPriceIncrementAmount": str(self.min_price_increment_amount),
            "initialMarginBuy": str(self.initial_margin_buy),
            "initialMarginSell": str(self.initial_margin_sell),
            "currency": self.currency,
        }

    @staticmethod
    def from_dict(data: dict, market: Optional[MarketKind] = None) -> "InstrumentMetadata":
        class_code = str(data.get("classCode", data.get("class_code", "")))
        market_value = data.get("market")
        if market_value:
            kind = MarketKind(market_value)
        elif market is not None:
            kind = market
        else:
            kind = MarketKind.FUTURES if class_code.upper() == "SPBFUT" else MarketKind.SPOT
        return InstrumentMetadata(
            instrument_uid=str(data.get("instrumentUid", data.get("instrument_uid", "")) or ""),
            ticker=str(data.get("ticker", "")),
            class_code=class_code,
            market=kind,
            figi=str(data.get("figi", "") or ""),
            name=str(data.get("name", "") or ""),
            lot_size=int(data.get("lotSize", data.get("lot_size", 1)) or 1),
            min_price_increment=Decimal(str(data.get("minPriceIncrement", "0.01"))),
            min_price_increment_amount=Decimal(str(data.get("minPriceIncrementAmount", "0"))),
            initial_margin_buy=Decimal(str(data.get("initialMarginBuy", "0"))),
            initial_margin_sell=Decimal(str(data.get("initialMarginSell", "0"))),
            currency=str(data.get("currency", "rub") or "rub"),
        )


@dataclass(frozen=True, slots=True)
class SpreadMetrics:
    """Метрики синтетического спреда пары на текущем тике.

    Статистика (``spread_rub``, ``mean_rub``, ``zscore``) хранится в ``float``:
    это «горячий» путь без аллокаций и без накладных расходов ``Decimal``.
    Цены мейкеров/микропрайсы остаются в ``Decimal`` — они идут в заявки.
    """

    time_ms: int
    spread_rub: float
    mean_rub: float
    variance: float
    zscore: float
    obi_leg1: float
    obi_leg2: float
    micro_leg1: Decimal
    micro_leg2: Decimal
    best_bid_leg1: Decimal
    best_ask_leg1: Decimal
    best_bid_leg2: Decimal
    best_ask_leg2: Decimal
    is_warm: bool
    tick_count: int
    beta: float = 1.0
    beta_macro: float = 0.0
    beta_drift: Optional[float] = None
    stat_sampled: bool = True

    @property
    def sigma_rub(self) -> float:
        return math.sqrt(max(self.variance, 1e-12))

    def expected_gross_reversion_rub(self, z_exit: float) -> float:
        """Ожидаемый валовой возврат к среднему в рублях на 1 лот пары."""
        deviation = abs(self.spread_rub - self.mean_rub)
        az = abs(self.zscore)
        if az <= 1e-9:
            return 0.0
        remaining = max(az - abs(z_exit), 0.0) / az
        return deviation * remaining


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """Завершённая круговая сделка (round-trip)."""

    trade_id: int
    opened_ms: int
    closed_ms: int
    topology: ExecutionTopology
    side_leg1: Side
    quantity: int
    leg1_entry_price: Decimal
    leg1_exit_price: Decimal
    leg2_entry_price: Optional[Decimal]
    leg2_exit_price: Optional[Decimal]
    gross_pnl_rub: Decimal
    fees_rub: Decimal
    net_pnl_rub: Decimal
    entry_z: float
    exit_z: float
    margin_rub: Decimal
    exit_reason: ExitReason
    mfe_rub: Decimal = ZERO
    mae_rub: Decimal = ZERO

    @property
    def hold_seconds(self) -> float:
        return max(0.0, (self.closed_ms - self.opened_ms) / 1000.0)

    @property
    def is_win(self) -> bool:
        return self.net_pnl_rub > 0


@dataclass(frozen=True, slots=True)
class DailyBacktestSummary:
    """Итоги торгового дня для отчёта бэктеста."""

    date: str
    trades: int
    wins: int
    win_rate_pct: float
    gross_pnl_rub: Decimal
    fees_rub: Decimal
    net_pnl_rub: Decimal
    net_pnl_pct_of_margin: float
    max_drawdown_rub: Decimal
    avg_hold_seconds: float


@dataclass(frozen=True, slots=True)
class BacktestTotals:
    """Сводные метрики прогона."""

    trades: int
    profitable_days: int
    total_days: int
    gross_pnl_rub: Decimal
    fees_rub: Decimal
    net_pnl_rub: Decimal
    win_rate_pct: float
    profit_factor: float
    sharpe_annual: float
    avg_hold_seconds: float
    fee_drag_pct: float
    max_drawdown_rub: Decimal


@dataclass(slots=True)
class OpenPosition:
    """Открытая позиция внутри конечного автомата."""

    trade_id: int
    opened_ms: int
    side_leg1: Side
    quantity: int
    leg1_entry_price: Decimal
    leg1_filled_qty: int = 0
    leg1_exit_price: Optional[Decimal] = None
    leg1_closed_qty: int = 0
    leg2_entry_price: Optional[Decimal] = None
    leg2_filled_qty: int = 0
    leg2_exit_price: Optional[Decimal] = None
    leg2_closed_qty: int = 0
    fees_rub: Decimal = ZERO
    entry_z: float = 0.0
    margin_rub: Decimal = ZERO
    exit_quote_started_ms: int = 0
    exit_deadline_ms: int = 0
    hedge_started_ms: int = 0
    frozen_mu: float = 0.0
    frozen_sigma: float = 0.0
    frozen_beta: float = 1.0
    entry_spread: float = 0.0
    round_trip_fee_rub: Decimal = ZERO
    tp_price: Optional[Decimal] = None
    leg1_flat_since_ms: int = 0
    scratch_until_ms: int = 0
    # --- v2.2: MFE/MAE, breakeven-ratchet, сжимаемая цель тейка ---
    max_favorable_gross_rub: Decimal = ZERO
    min_adverse_gross_rub: Decimal = ZERO
    breakeven_locked: bool = False
    tp_target_rub: Decimal = ZERO
    tags: dict = field(default_factory=dict)

    def add_fee(self, fee: Decimal) -> None:
        self.fees_rub += fee
