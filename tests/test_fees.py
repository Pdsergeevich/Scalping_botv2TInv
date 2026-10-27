"""Проверки прогрессивного тарифного движка и защиты от комиссии."""

from datetime import date
from decimal import Decimal

from bot_a.economics.fees import Tariff, TinkoffFeeCalculator
from bot_a.models.domain import ExecutionTopology
from tests.helpers import make_meta, make_spot


def test_trader_tier_progression():
    calc = TinkoffFeeCalculator(Tariff.TRADER)
    assert calc.current_rate == Decimal("0.0004")
    first = calc.register_notional(Decimal("5000000"))
    assert first == Decimal("2000.00")
    assert calc.current_rate == Decimal("0.0004")  # ровно на границе — ещё первая ступень
    calc.register_notional(Decimal("1"))
    assert calc.current_rate == Decimal("0.0003")
    calc.register_notional(Decimal("5000000"))
    assert calc.current_rate == Decimal("0.00025")


def test_premium_tier_progression():
    calc = TinkoffFeeCalculator(Tariff.PREMIUM)
    assert calc.current_rate == Decimal("0.00025")
    calc.register_notional(Decimal("5000001"))
    assert calc.current_rate == Decimal("0.0002")
    calc.register_notional(Decimal("12000000"))
    assert calc.current_rate == Decimal("0.00015")


def test_round_trip_fee_counts_sides_by_topology():
    calc = TinkoffFeeCalculator(Tariff.TRADER)
    leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01")
    leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01")
    single = calc.estimated_round_trip_fee(
        leg1, Decimal("100"), 1, leg2, Decimal("100"), ExecutionTopology.SINGLE_LEG_LEAD_LAG
    )
    dual = calc.estimated_round_trip_fee(
        leg1, Decimal("100"), 1, leg2, Decimal("100"), ExecutionTopology.DUAL_LEG_HEDGED
    )
    assert single == Decimal("0.08")
    assert dual == Decimal("0.16")


def test_fee_accumulates_turnover_and_rolls_day():
    calc = TinkoffFeeCalculator(Tariff.TRADER)
    calc.register_notional(Decimal("6000000"))
    assert calc.daily_turnover_rub == Decimal("6000000")
    assert calc.daily_fees_rub == Decimal("2400")
    calc.roll_day(date(2025, 1, 2))
    assert calc.daily_turnover_rub == Decimal("0")


def test_edge_guard():
    assert TinkoffFeeCalculator.is_edge_sufficient(
        Decimal("10"), Decimal("1"), Decimal("1"), Decimal("2")
    )
    assert not TinkoffFeeCalculator.is_edge_sufficient(
        Decimal("3.9"), Decimal("1"), Decimal("1"), Decimal("2")
    )
    assert TinkoffFeeCalculator.is_edge_sufficient(Decimal("0"), Decimal("0"), Decimal("0"), Decimal("100"))


def test_expected_gross_reversion():
    value = TinkoffFeeCalculator.expected_gross_reversion_rub(10.0, 0.0, 4.0, 1.0)
    assert value == Decimal("7.5")
    assert TinkoffFeeCalculator.expected_gross_reversion_rub(0.0, 0.0, 0.0, 0.0) == Decimal("0")


def test_slippage_buffer_for_futures_and_spot():
    futures = make_meta("AAA", price_step="0.01", step_amount="1")
    assert TinkoffFeeCalculator.slippage_buffer_rub(futures, 2, 3) == Decimal("6")
    spot = make_meta("SBER", price_step="0.01", step_amount="0", futures=False)
    assert TinkoffFeeCalculator.slippage_buffer_rub(spot, 2, 1) == Decimal("0.02")
    spot_lot = make_spot("SBERP", lot=10)
    # Для акции шаг 0.01 RUB на лот из 10 акций стоит 0.1 RUB.
    assert TinkoffFeeCalculator.slippage_buffer_rub(spot_lot, 2, 1) == Decimal("0.2")
