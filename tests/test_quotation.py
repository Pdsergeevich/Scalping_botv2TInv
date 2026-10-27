"""Проверки точных преобразований котировок и экономики пункта цены."""

from decimal import Decimal

from bot_a.models.domain import InstrumentMetadata, MarketKind
from bot_a.utils.quotation import (
    ceil_to_increment,
    dec,
    decimal_to_quotation,
    floor_to_increment,
    quotation_from_proto,
    quotation_to_decimal,
    round_to_increment,
    shift_by_steps,
    steps_between,
)


def test_quotation_basic_conversion():
    assert quotation_to_decimal(1, 500_000_000) == Decimal("1.5")
    assert quotation_to_decimal(-2, -250_000_000) == Decimal("-2.25")
    assert quotation_to_decimal(0, 1) == Decimal("0.000000001")


def test_quotation_round_trip_positive_and_negative():
    for value in ("0", "1.2345", "-1.2345", "7.2930", "-0.001", "1000000.000000001"):
        units, nano = decimal_to_quotation(Decimal(value))
        assert quotation_to_decimal(units, nano) == Decimal(value)


def test_decimal_to_quotation_signs():
    units, nano = decimal_to_quotation(Decimal("-1.5"))
    assert units == -1
    assert nano == -500_000_000


def test_quotation_from_proto_accepts_objects():
    class Proto:
        units = 3
        nano = 250_000_000

    assert quotation_from_proto(Proto()) == Decimal("3.25")
    assert quotation_from_proto((2, 500_000_000)) == Decimal("2.5")
    assert quotation_from_proto(Decimal("9.99")) == Decimal("9.99")


def test_increment_rounding():
    assert round_to_increment(Decimal("100.015"), Decimal("0.01")) == Decimal("100.02")
    assert floor_to_increment(Decimal("100.019"), Decimal("0.01")) == Decimal("100.01")
    assert ceil_to_increment(Decimal("100.011"), Decimal("0.01")) == Decimal("100.02")
    assert steps_between(Decimal("100.00"), Decimal("100.05"), Decimal("0.01")) == 5
    assert steps_between(Decimal("100.05"), Decimal("100.00"), Decimal("0.01")) == -5
    assert shift_by_steps(Decimal("100.00"), 3, Decimal("0.01")) == Decimal("100.03")


def test_futures_point_to_ruble_math():
    meta = InstrumentMetadata(
        instrument_uid="uid",
        ticker="CNYRUBF",
        class_code="SPBFUT",
        market=MarketKind.FUTURES,
        min_price_increment=Decimal("0.001"),
        min_price_increment_amount=Decimal("1"),
        initial_margin_buy=Decimal("9000"),
        initial_margin_sell=Decimal("9500"),
    )
    assert meta.rub_per_price_unit == Decimal("1000")
    assert meta.value_in_rub(Decimal("7.2")) == Decimal("7200.0")
    assert meta.position_margin_rub(2) == Decimal("19000")
    assert meta.price_steps(Decimal("7.200"), Decimal("7.205")) == 5


def test_spot_lot_math():
    meta = InstrumentMetadata(
        instrument_uid="uid",
        ticker="SBERP",
        class_code="TQBR",
        market=MarketKind.SPOT,
        lot_size=10,
        min_price_increment=Decimal("0.01"),
    )
    assert meta.rub_per_price_unit == Decimal("10")
    assert meta.value_in_rub(Decimal("300.50")) == Decimal("3005.0")
    assert meta.position_margin_rub(0) == Decimal("0")


def test_dec_helper_is_deterministic_for_floats():
    assert dec(0.1) + dec(0.2) == Decimal("0.3")
