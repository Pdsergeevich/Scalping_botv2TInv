"""Точные преобразования котировок T-Invest и работа с шагом цены.

Все расчёты денег выполняются в ``Decimal``: бинарные ``float`` недопустимы
там, где сравниваются цены заявок и считаются комиссии.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Any, Tuple

NANO_SCALE = Decimal(1_000_000_000)
NANO_QUANT = Decimal("0.000000001")


def dec(value: Any) -> Decimal:
    """Безопасное приведение к ``Decimal`` (float конвертируется через ``str``)."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return Decimal(int(value))
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    if value is None:
        return Decimal(0)
    return Decimal(str(value))


def quotation_to_decimal(units: int, nano: int) -> Decimal:
    """``Quotation(units, nano)`` -> ``Decimal``."""
    return Decimal(int(units)) + Decimal(int(nano)) / NANO_SCALE


def decimal_to_quotation(value: Any) -> Tuple[int, int]:
    """``Decimal`` -> ``(units, nano)`` с одинаковым знаком обеих частей."""
    amount = abs(dec(value)).quantize(NANO_QUANT, rounding=ROUND_HALF_UP)
    units = int(amount)
    nano = int((amount - Decimal(units)) * NANO_SCALE)
    if dec(value) < 0:
        return -units, -nano
    return units, nano


def quotation_from_proto(value: Any) -> Decimal:
    """Принимает protobuf ``Quotation``/``MoneyValue``, tuple, число или ``Decimal``."""
    if value is None:
        return Decimal(0)
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return dec(value)
    if isinstance(value, tuple) and len(value) >= 2:
        return quotation_to_decimal(value[0], value[1])
    units = getattr(value, "units", None)
    if units is not None:
        return quotation_to_decimal(int(units), int(getattr(value, "nano", 0)))
    return dec(value)


def money_value_to_decimal(value: Any) -> Decimal:
    """``MoneyValue`` -> ``Decimal`` (валюта отбрасывается)."""
    return quotation_from_proto(value)


def decimal_to_money(value: Any) -> Tuple[int, int]:
    return decimal_to_quotation(value)


def round_to_increment(price: Any, increment: Any, rounding: str = ROUND_HALF_UP) -> Decimal:
    """Округление цены до шага ``increment`` (по умолчанию — к ближайшему)."""
    p = dec(price)
    inc = dec(increment)
    if inc <= 0:
        return p
    return (p / inc).to_integral_value(rounding=rounding) * inc


def floor_to_increment(price: Any, increment: Any) -> Decimal:
    return round_to_increment(price, increment, ROUND_FLOOR)


def ceil_to_increment(price: Any, increment: Any) -> Decimal:
    return round_to_increment(price, increment, ROUND_CEILING)


def steps_between(price_from: Any, price_to: Any, increment: Any) -> int:
    """Количество шагов цены между двумя ценами (знаковое)."""
    inc = dec(increment)
    if inc <= 0:
        return 0
    return int(((dec(price_to) - dec(price_from)) / inc).to_integral_value(rounding=ROUND_HALF_UP))


def shift_by_steps(price: Any, steps: int, increment: Any) -> Decimal:
    return dec(price) + dec(increment) * Decimal(int(steps))



def fmt_rub(value: Any, places: int = 2) -> str:
    """Формат рублей для отчётов: ``1 234 567.89``."""
    amount = dec(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    text = f"{amount:,.{places}f}"
    return text.replace(",", " ")




