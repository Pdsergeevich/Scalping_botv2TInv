"""Тесты v2.11: починка оценки беты, гейт входов, пересев EWMA, потолок тейка, дневной лимит.

Корень патча — отчёт `report_20261105.md` §4: в `PairSpreadCalculator.on_book` флаг
`pending` сбрасывался на любом событии с обеими ногами, а `_update_beta` вызывался
только если это же событие оказалось статистическим сэмплом. При синхронных парах
событий (прогрев ISS, `VTBR_SBER`) наблюдение терялось всегда: **1 из 41 940** тиков.
Итог живого дня: бета осталась статической (`1.0`), `mu` = 235 RUB при реальном спреде
1 RUB, `|Z|` = 34.5, тейк на **9733 шага** (−88 % цены), `−8.67 ₽` и блокировка пары.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from bot_a.math_engine.pair_spread import PairSpreadCalculator
from bot_a.models.domain import ExecutionTopology, ExitReason
from tests.helpers import make_book, make_meta
from tests.test_engine import _bundle, _open_position, _params

SINGLE = ExecutionTopology.SINGLE_LEG_LEAD_LAG


def _calc(**kwargs) -> PairSpreadCalculator:
    base = dict(
        leg1=make_meta("AAA", uid="uid-AAA"),
        leg2=make_meta("BBB", uid="uid-BBB"),
        beta=1.0,
        ewma_window=20,
        beta_mode="slow_ratio",
        beta_min_samples=5,
        macro_beta_window=200,
        macro_beta_min_samples=10,
        stat_sample_interval_seconds=5.0,
    )
    base.update(kwargs)
    return PairSpreadCalculator(**base)


def _feed_pair(calc, time_ms: int, price1: str = "100.00", price2: str = "500.00"):
    """Синхронная пара событий с ОДНИМ таймстемпом — как в прогреве ISS (VTBR)."""
    calc.on_book(0, make_book(Decimal(price1), time_ms, uid="uid-AAA"))
    return calc.on_book(1, make_book(Decimal(price2), time_ms, uid="uid-BBB"))


def test_synchronised_pairs_estimate_beta_every_sample():
    """P1: синхронные пары не теряют наблюдения — по одному на каждый 5-секундный сэмпл."""
    calc = _calc()
    base = 1_700_000_000_000
    samples = 60
    for index in range(samples):
        metrics = _feed_pair(calc, base + index * 5000, "100.00", "500.00")
    assert metrics is not None
    assert calc.beta_observations >= 30, calc.beta_observations
    assert calc.beta_estimated is True
    # β = V2 / V1 = 500 / 100 = 5 — статическая 1.0 больше не используется
    assert 4.0 < calc.beta < 6.5, calc.beta


def test_beta_observations_grow_one_per_sample_not_per_tick():
    """P1: на каждый сэмпл приходится ровно одно наблюдение (без дублей и потерь)."""
    calc = _calc()
    base = 1_700_000_000_000
    for index in range(20):
        _feed_pair(calc, base + index * 5000, "100.00", "500.00")
        # внутри 5-секундного окна сыпем тики — они не должны двигать бету
        for offset in range(1, 5):
            _feed_pair(calc, base + index * 5000 + offset * 100, "100.00", "500.00")
    assert calc.beta_observations == 20, calc.beta_observations


def test_beta_estimated_false_before_min_samples():
    calc = _calc()
    base = 1_700_000_000_000
    for index in range(3):
        _feed_pair(calc, base + index * 5000)
    assert calc.beta_estimated is False
    assert calc.beta_observations < 5


def test_static_mode_is_always_estimated():
    """Для режимов без оценки (`static`) гейт входов не включается."""
    calc = _calc(beta_mode="static", beta=3.0)
    assert calc.beta_estimated is True


def test_first_beta_estimate_reseeds_ewma():
    """P3: при первой оценке беты `mu`/`variance` пересчитываются, а не «ползут» месяцами."""
    calc = _calc(beta_min_samples=10)
    base = 1_700_000_000_000
    # 9 сэмплов с β = 1.0: спред ≈ 500 − 100 = 400 RUB
    for index in range(9):
        metrics = _feed_pair(calc, base + index * 5000, "100.00", "500.00")
    assert calc.beta_estimated is False
    assert metrics.mean_rub > 300.0, metrics.mean_rub
    # десятый сэмпл оценивает бету как 5.0 → спред должен немедленно стать ≈ 0
    for index in range(9, 40):
        metrics = _feed_pair(calc, base + index * 5000, "100.00", "500.00")
    assert calc.beta_estimated is True
    assert abs(calc.beta - 5.0) < 1e-3, calc.beta
    assert abs(metrics.mean_rub) < 1.0, f"mu не пересеяна: {metrics.mean_rub}"
    assert calc.beta_reseeds >= 1
    # σ в правильном масштабе (единицы рублей, а не сотни)
    assert metrics.variance ** 0.5 < 5.0, metrics.variance ** 0.5


def test_big_beta_jump_reseeds_ewma():
    """P3: скачок беты >20 % тоже пересевает EWMA (новая история в новом масштабе)."""
    calc = _calc(beta_min_samples=10)
    base = 1_700_000_000_000
    for index in range(60):
        _feed_pair(calc, base + index * 5000, "100.00", "500.00")
    before = calc.beta_reseeds
    # Скачок 24 %: 5.0 → 6.2 (как «догон» беты в живом VTBR 05.11)
    calc._set_beta(6.2)  # noqa: SLF001
    assert calc.beta == 6.2
    assert calc.beta_reseeds > before, "скачок беты обязан пересеять EWMA"
    # μ пересчитана с новой бетой: 500 − 6.2×100 = −120, а не 400
    assert abs(calc.ewma.mean - (-120.0)) < 1.0, calc.ewma.mean
    # меньше 20 % — пересева нет
    before = calc.beta_reseeds
    calc._set_beta(6.5)  # noqa: SLF001
    assert calc.beta_reseeds == before


def test_entry_blocked_without_estimated_beta(tmp_path):
    """P2: пока бета не оценена — входы запрещены и считается betaNotEstimatedSkips."""

    async def scenario():
        params = _params(beta_mode="slow_ratio", beta_min_samples=10_000)
        config, clock, _fee, gateway, engine = _bundle(tmp_path, SINGLE, params)
        await gateway.start()
        # Прогрев статистики без права входа (гейт беты режет всё).
        for _ in range(80):
            clock.advance(1000)
            await engine.on_clock()
            book1 = make_book(Decimal("100.00"), clock.now_ms(), uid="uid-AAA")
            book2 = make_book(Decimal("104.00"), clock.now_ms(), uid="uid-BBB")
            await engine.on_order_book(1, book1)
            await engine.on_order_book(2, book2)
            await engine.on_trade(1, _trade(book1, clock.now_ms()))
            await engine.on_trade(2, _trade(book2, clock.now_ms()))
        assert engine.position is None
        assert engine.counters["betaNotEstimatedSkips"] > 0
        assert engine.counters["entries"] == 0

    asyncio.run(scenario())


def test_entry_allowed_when_beta_estimated(tmp_path):
    """P2: как только бета оценена, гейт не мешает обычному входу."""

    async def scenario():
        params = _params(beta_mode="slow_ratio", beta_min_samples=5)
        config, clock, _fee, gateway, engine = _bundle(tmp_path, SINGLE, params)
        await gateway.start()
        await _open_position(engine, clock, gateway)
        return engine

    engine = asyncio.run(scenario())
    assert engine.position is not None
    assert engine.beta_estimated is True


def _trade(book, time_ms):
    from tests.helpers import make_trade

    return make_trade(book.bids[0].price, time_ms, quantity=50, uid=book.instrument_uid)


def test_tp_absolute_ceiling_is_always_applied(tmp_path):
    """P4: 9733 шага невозможны — абсолютный потолок работает при любой бете."""

    async def scenario():
        params = _params(
            tp_max_sigma_multiple=4.5,
            tp_max_notional_fraction=0.012,
            tp_tight_sigma_multiple=0.0,
            tp_tight_notional_fraction=0.0,
            tp_cap_beta_threshold=0.8,
        )
        config, clock, _fee, gateway, engine = _bundle(tmp_path, SINGLE, params)
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        assert position is not None
        engine._freeze_position(position, engine.metrics)  # noqa: SLF001
        # Имитируем катастрофу 05.11: mu уехала на 220 RUB от спреда.
        position.frozen_mu = position.entry_spread - 220.0
        position.frozen_beta = 5.25
        target = engine._target_tp_rub(position, clock.now_ms())  # noqa: SLF001
        price = engine._tp_price_for(position, target)  # noqa: SLF001
        assert price is not None
        steps = abs(price - position.leg1_entry_price) / engine._leg1.min_price_increment  # noqa: SLF001
        notional = position.leg1_entry_price * engine._leg1.rub_per_price_unit  # noqa: SLF001
        assert target <= notional * Decimal("0.012") + Decimal("1e-9")
        assert steps < Decimal("200"), f"тейк уехал на {steps} шагов"

    asyncio.run(scenario())


def test_daily_limit_uses_sigma_floor(tmp_path):
    """P5: дневной лимит = max(3.5 × fee, 1.5 × σ_Leg1,позиции) — не 5.32 RUB на 55 лотах."""

    async def scenario(suffix: str, loss_sigmas: Decimal):
        params = _params(
            use_daily_stop_breaker=True,
            max_daily_loss_fee_multiple=Decimal("3.5"),
            daily_stop_sigma_multiple=1.5,
        )
        config, clock, _fee, gateway, engine = _bundle(tmp_path / suffix, SINGLE, params)
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        assert position is not None
        engine._freeze_position(position, engine.metrics)  # noqa: SLF001
        lots = max(position.leg1_filled_qty, 1)
        sigma_pos = engine._sigma_leg1_rub_of(position) * Decimal(lots)  # noqa: SLF001
        fee = engine._current_round_trip_fee(lots)  # noqa: SLF001
        assert sigma_pos > fee, f"σ позиции {sigma_pos} должна быть больше комиссии {fee}"
        engine._register_daily_outcome(  # noqa: SLF001
            ExitReason.STAGNATION, -(sigma_pos * loss_sigmas), clock.now_ms(), sigma_pos
        )
        return engine, fee, sigma_pos

    # 1.2 σ — по старому правилу (3.5 × fee = маленькое число) пара была бы заблокирована
    engine, fee, sigma_pos = asyncio.run(scenario("a", Decimal("1.2")))
    assert fee * Decimal("3.5") < sigma_pos * Decimal("1.2"), "сценарий должен быть показательным"
    assert engine.daily_locked_reason is None, "σ-пол не даёт заблокировать пару на 1.2 σ"

    # 5 σ — реальный крупный убыток по-прежнему блокирует пару
    engine2, _fee2, _sigma2 = asyncio.run(scenario("b", Decimal("5")))
    assert engine2.daily_locked_reason is not None, "крупный убыток обязан блокировать пару"


def test_pair_label_is_used_in_logs(tmp_path, caplog):
    """P5: в логах видно, чья строка — `AAA/BBB`."""
    import logging

    async def scenario():
        params = _params(beta_mode="slow_ratio", beta_min_samples=10_000)
        config, clock, _fee, gateway, engine = _bundle(tmp_path, SINGLE, params)
        await gateway.start()
        assert engine.pair_label == "AAA/BBB"
        book1 = make_book(Decimal("100.00"), clock.now_ms(), uid="uid-AAA")
        book2 = make_book(Decimal("104.00"), clock.now_ms(), uid="uid-BBB")
        await engine.on_order_book(1, book1)
        await engine.on_order_book(2, book2)
        await engine._maybe_enter(engine.metrics)  # noqa: SLF001

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())
    assert any("AAA/BBB" in record.getMessage() for record in caplog.records), [
        r.getMessage() for r in caplog.records
    ]
