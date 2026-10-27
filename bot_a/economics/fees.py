"""Прогрессивный тарифный движок T-Invest и защита от «съедающей» комиссии.

Брокер берёт процент от полной рублёвой стоимости контракта с каждой стороны
сделки (и мейкер, и тейкер). Ставка зависит от накопленного оборота за день,
поэтому расчёт ведётся инкрементально тем же объектом, который обслуживает
и живой режим, и бэктест.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple

from bot_a.models.domain import ExecutionTopology, InstrumentMetadata
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import TimeProvider, trading_day_key

ZERO = Decimal("0")


class Tariff(str, Enum):
    """Тарифный план T-Invest."""

    TRADER = "Trader"
    PREMIUM = "Premium"


@dataclass(frozen=True, slots=True)
class FeeTier:
    """Ступень тарифа: ставка действует, пока дневной оборот ``<= upto_rub``."""

    upto_rub: Optional[Decimal]
    rate: Decimal


# Значения строго по условиям задачи (2.1).
DEFAULT_TIERS: dict[Tariff, Tuple[FeeTier, ...]] = {
    Tariff.TRADER: (
        FeeTier(Decimal("5000000"), Decimal("0.0004")),
        FeeTier(Decimal("10000000"), Decimal("0.0003")),
        FeeTier(None, Decimal("0.00025")),
    ),
    Tariff.PREMIUM: (
        FeeTier(Decimal("5000000"), Decimal("0.00025")),
        FeeTier(Decimal("17000000"), Decimal("0.0002")),
        FeeTier(None, Decimal("0.00015")),
    ),
}


class TinkoffFeeCalculator:
    """Учёт дневного оборота и расчёт комиссий по ступеням тарифа."""

    __slots__ = ("_tariff", "_tiers", "_clock", "_turnover_rub", "_day", "_fills", "_fees_rub")

    def __init__(
        self,
        tariff: Tariff = Tariff.TRADER,
        tiers: Optional[dict[Tariff, Tuple[FeeTier, ...]]] = None,
        clock: Optional[TimeProvider] = None,
    ) -> None:
        self._tariff = tariff
        self._tiers = tiers or DEFAULT_TIERS
        self._clock = clock
        self._turnover_rub = ZERO
        self._fees_rub = ZERO
        self._fills = 0
        self._day: Optional[date] = trading_day_key(clock.now_ms()) if clock else None

    # ------------------------------------------------------------------ состояние
    @property
    def tariff(self) -> Tariff:
        return self._tariff

    @property
    def daily_turnover_rub(self) -> Decimal:
        return self._turnover_rub

    @property
    def daily_fees_rub(self) -> Decimal:
        return self._fees_rub

    @property
    def fills(self) -> int:
        return self._fills

    @property
    def tiers(self) -> Tuple[FeeTier, ...]:
        return self._tiers[self._tariff]

    @property
    def current_rate(self) -> Decimal:
        """Действующая ставка за одну сторону сделки."""
        for tier in self.tiers:
            if tier.upto_rub is None or self._turnover_rub <= tier.upto_rub:
                return tier.rate
        return self.tiers[-1].rate

    @property
    def tier_index(self) -> int:
        for index, tier in enumerate(self.tiers):
            if tier.upto_rub is None or self._turnover_rub <= tier.upto_rub:
                return index
        return len(self.tiers) - 1

    def reset_turnover(self) -> None:
        self._turnover_rub = ZERO
        self._fees_rub = ZERO
        self._fills = 0

    def roll_day(self, day: Optional[date] = None) -> None:
        """Сброс оборота при смене торгового дня."""
        if day is None and self._clock is not None:
            day = trading_day_key(self._clock.now_ms())
        if day is not None and self._day != day:
            self._day = day
            self.reset_turnover()

    def _ensure_day(self) -> None:
        if self._clock is not None:
            self.roll_day(trading_day_key(self._clock.now_ms()))

    # ------------------------------------------------------------------ расчёты
    def fee_for_notional(self, notional_rub: Decimal) -> Decimal:
        """Комиссия за одну сторону при текущей ступени (без изменения состояния)."""
        return dec(notional_rub) * self.current_rate

    def register_notional(self, notional_rub: Decimal) -> Decimal:
        """Регистрирует сторону сделки: возвращает списанную комиссию."""
        self._ensure_day()
        amount = abs(dec(notional_rub))
        fee = amount * self.current_rate
        self._turnover_rub += amount
        self._fees_rub += fee
        self._fills += 1
        return fee

    def register_fill(self, price: Decimal, quantity: int, meta: InstrumentMetadata) -> Decimal:
        """Комиссия по факту исполнения: цена в пунктах/рублях, количество в лотах."""
        notional = meta.value_in_rub(price) * abs(int(quantity))
        return self.register_notional(notional)

    def estimate_sequence(self, notionals: Sequence[Decimal]) -> Decimal:
        """Суммарная комиссия по последовательности сторон с учётом роста оборота."""
        turnover = self._turnover_rub
        total = ZERO
        tiers = self.tiers
        for notional in notionals:
            amount = abs(dec(notional))
            rate = tiers[-1].rate
            for tier in tiers:
                if tier.upto_rub is None or turnover <= tier.upto_rub:
                    rate = tier.rate
                    break
            total += amount * rate
            turnover += amount
        return total

    def estimated_round_trip_fee(
        self,
        leg1: InstrumentMetadata,
        price1: Decimal,
        quantity: int,
        leg2: Optional[InstrumentMetadata] = None,
        price2: Optional[Decimal] = None,
        topology: ExecutionTopology = ExecutionTopology.DUAL_LEG_HEDGED,
    ) -> Decimal:
        """Комиссия полного круга: 2 стороны на ногу (SingleLeg) или 4 (DualLeg)."""
        qty = abs(int(quantity))
        notionals = [leg1.value_in_rub(price1) * qty] * 2
        if topology is ExecutionTopology.DUAL_LEG_HEDGED and leg2 is not None and price2 is not None:
            notionals.extend([leg2.value_in_rub(price2) * qty] * 2)
        return self.estimate_sequence(notionals)

    def round_trip_fee_of_notionals(self, notionals: Iterable[Decimal]) -> Decimal:
        return self.estimate_sequence(list(notionals))

    # ------------------------------------------------------------------ защита от комиссии
    @staticmethod
    def expected_gross_reversion_rub(spread_rub: float, mean_rub: float, zscore: float, z_exit: float) -> Decimal:
        """Ожидаемый валовой возврат ``|S - mu| * (|Z| - Z_exit)/|Z|`` в рублях."""
        az = abs(zscore)
        if az <= 1e-9:
            return ZERO
        remaining = max(az - abs(z_exit), 0.0) / az
        return dec(abs(spread_rub - mean_rub) * remaining)

    @staticmethod
    def slippage_buffer_rub(meta: InstrumentMetadata, steps: int, quantity: int) -> Decimal:
        """Рублёвый буфер проскальзывания на ``steps`` шагов цены.

        Для фьючерса шаг стоит ``MinPriceIncrementAmount``, для акции —
        ``MinPriceIncrement * LotSize`` (цена шага умножается на размер лота).
        """
        if meta.is_futures:
            step_value = meta.min_price_increment_amount
        else:
            step_value = meta.min_price_increment * Decimal(meta.lot_size or 1)
        if step_value <= 0:
            step_value = meta.min_price_increment
        return step_value * Decimal(max(int(steps), 0)) * Decimal(abs(int(quantity)))

    @staticmethod
    def total_round_trip_cost(fee_rub: Decimal, slippage_rub: Decimal) -> Decimal:
        return dec(fee_rub) + dec(slippage_rub)

    @staticmethod
    def is_edge_sufficient(
        expected_gross_reversion_rub: Decimal,
        round_trip_fee_rub: Decimal,
        slippage_buffer_rub: Decimal = ZERO,
        min_edge_multiplier: Decimal = Decimal("1.8"),
    ) -> bool:
        """Сигнал валиден, только если ожидаемый возврат покрывает издержки."""
        cost = TinkoffFeeCalculator.total_round_trip_cost(round_trip_fee_rub, slippage_buffer_rub)
        if cost <= 0:
            return True
        return dec(expected_gross_reversion_rub) >= dec(min_edge_multiplier) * cost

    def snapshot(self) -> dict:
        return {
            "tariff": self._tariff.value,
            "tier": self.tier_index + 1,
            "rate": str(self.current_rate),
            "dailyTurnoverRub": str(self._turnover_rub),
            "dailyFeesRub": str(self._fees_rub),
            "fills": self._fills,
        }
