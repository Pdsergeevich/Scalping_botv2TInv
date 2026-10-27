"""Проверки математического ядра: OBI, микропрайс, EWMA Z-score, спред пары."""

import math
from decimal import Decimal

from bot_a.math_engine.pair_spread import EwmaZScore, PairSpreadCalculator, book_stats, obi_weights
from tests.helpers import make_book, make_spot


def test_obi_weights_start_at_one():
    weights = obi_weights(5)
    assert weights[0] == 1.0
    assert math.isclose(weights[1], math.exp(-0.5))
    assert len(weights) == 5


def test_book_stats_balanced_book():
    book = make_book("100", 1000, volume=100, obi=0.0)
    stats = book_stats(book)
    assert stats is not None
    assert stats.best_bid == Decimal("100.00")
    assert stats.best_ask == Decimal("100.01")
    assert abs(stats.obi) < 1e-12
    assert math.isclose(stats.microprice, 100.005, abs_tol=1e-9)


def test_book_stats_bid_heavy_gives_positive_obi_and_higher_microprice():
    book = make_book("100", 1000, volume=1000, obi=0.6)
    stats = book_stats(book)
    assert stats is not None
    assert stats.obi > 0.5
    assert stats.microprice > 100.005  # смещение к аску


def test_book_stats_returns_none_for_one_sided_book():
    from bot_a.models.domain import OrderBookLevel, OrderBookSnapshot

    book = OrderBookSnapshot(
        instrument_uid="x",
        time_ms=1,
        bids=(OrderBookLevel(Decimal("100"), 10),),
        asks=(),
    )
    assert book_stats(book) is None


def test_ewma_zscore_warmup_and_reaction():
    ewma = EwmaZScore(window=10)
    for _ in range(20):
        mean, variance, zscore, warm = ewma.update(100.0)
    assert warm
    assert variance == 0.0
    assert zscore == 0.0
    mean, variance, zscore, warm = ewma.update(105.0)
    assert warm
    assert zscore > 1.0
    assert variance > 0.0


def test_pair_spread_math_and_beta():
    leg1 = make_spot("AAA")
    leg2 = make_spot("BBB")
    calc = PairSpreadCalculator(leg1, leg2, beta=1.0, ewma_window=5)
    assert calc.on_book(0, make_book("100", 1000)) is None
    metrics = calc.on_book(1, make_book("101", 1001))
    assert metrics is not None
    assert math.isclose(metrics.spread_rub, 1.0, abs_tol=1e-9)
    assert metrics.is_warm is False
    assert metrics.obi_leg1 == 0.0
    assert metrics.tick_count == 1

    for index in range(10):
        calc.on_book(0, make_book("100", 2000 + index))
        metrics = calc.on_book(1, make_book("101", 2000 + index))
    assert metrics is not None and metrics.is_warm

    metrics = calc.on_book(1, make_book("104", 9000))
    assert metrics is not None
    assert metrics.zscore > 1.0
    assert metrics.spread_rub > 3.0


def test_pair_spread_beta_scales_leg1():
    leg = make_spot("AAA")
    leg2 = make_spot("BBB")
    calc = PairSpreadCalculator(leg, leg2, beta=2.0, ewma_window=5, use_dynamic_beta=False)
    calc.on_book(0, make_book("100", 1))
    metrics = calc.on_book(1, make_book("201", 2))
    assert metrics is not None
    assert math.isclose(metrics.spread_rub, 0.995, abs_tol=1e-9)
    assert metrics.beta == 2.0


def test_dynamic_beta_converges_to_volatility_ratio():
    leg1 = make_spot("AAA")
    leg2 = make_spot("BBB")
    calc = PairSpreadCalculator(
        leg1,
        leg2,
        beta=1.0,
        ewma_window=20,
        use_dynamic_beta=True,
        beta_min=0.1,
        beta_max=20.0,
        beta_min_samples=10,
    )
    price1 = Decimal("100")
    price2 = Decimal("200")
    last = None
    for index in range(300):
        delta = Decimal("0.01") if index % 2 == 0 else Decimal("-0.01")
        price1 += delta
        price2 += delta * 3
        calc.on_book(0, make_book(price1, index * 2))
        last = calc.on_book(1, make_book(price2, index * 2 + 1))
    assert last is not None
    assert 2.5 < calc.beta < 3.5, calc.beta
    assert last.beta == calc.beta


def test_dynamic_beta_can_be_disabled():
    leg1 = make_spot("AAA")
    leg2 = make_spot("BBB")
    calc = PairSpreadCalculator(
        leg1, leg2, beta=1.0, ewma_window=20, use_dynamic_beta=False, beta_min_samples=5
    )
    price1 = Decimal("100")
    price2 = Decimal("200")
    for index in range(50):
        delta = Decimal("0.01") if index % 2 == 0 else Decimal("-0.01")
        price1 += delta
        price2 += delta * 3
        calc.on_book(0, make_book(price1, index * 2))
        calc.on_book(1, make_book(price2, index * 2 + 1))
    assert calc.beta == 1.0


def test_slow_ratio_beta_matches_value_ratio_and_freezes():
    leg1 = make_spot("AAA")
    leg2 = make_spot("BBB")
    calc = PairSpreadCalculator(
        leg1,
        leg2,
        beta=1.0,
        ewma_window=40,
        beta_mode="slow_ratio",
        beta_min_samples=10,
    )
    price1 = Decimal("100")
    price2 = Decimal("200")
    timestamp = 1_000_000
    for index in range(300):
        delta = Decimal("0.01") if index % 2 == 0 else Decimal("-0.01")
        price1 += delta
        price2 += delta * 2
        timestamp += 2
        calc.on_book(0, make_book(price1, timestamp))
        calc.on_book(1, make_book(price2, timestamp + 1))
    assert abs(calc.beta - 2.0) < 0.05, calc.beta

    calc.freeze_beta()
    frozen = calc.beta
    for index in range(40):
        price1 += Decimal("0.05")
        price2 -= Decimal("0.05")
        timestamp += 2
        calc.on_book(0, make_book(price1, timestamp))
        calc.on_book(1, make_book(price2, timestamp + 1))
    assert calc.beta == frozen, "бета обязана быть неизменной при открытой позиции"

    calc.unfreeze_beta()
    for index in range(200):
        price1 += Decimal("0.005")
        price2 += Decimal("0.02")
        timestamp += 2
        calc.on_book(0, make_book(price1, timestamp))
        calc.on_book(1, make_book(price2, timestamp + 1))
    assert calc.beta != frozen


def test_same_underlying_beta_is_constant():
    leg1 = make_spot("AAA")
    leg2 = make_spot("BBB")
    calc = PairSpreadCalculator(
        leg1, leg2, beta=1.0, ewma_window=20, beta_mode="same_underlying", beta_min_samples=5
    )
    price1 = Decimal("100")
    price2 = Decimal("300")
    for index in range(50):
        price1 += Decimal("0.02")
        price2 -= Decimal("0.03")
        calc.on_book(0, make_book(price1, index * 2))
        calc.on_book(1, make_book(price2, index * 2 + 1))
    assert calc.beta == 1.0
