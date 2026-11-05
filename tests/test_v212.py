"""Тесты v2.12: медленный нож (3 мин в шагах/σ) и гейт зрелости σ после пересева беты.

Основание — отчёт `l2_recordings_benchmark_v2_11.md` §8.1/8.2:

* `VTBR_SBER` 02.11 11:37:21 `BUY` (`−19.42 RUB`, худшая сделка окна) — `Leg1` прошла
  против сделки **21 шаг = 1.25 σ** ровным темпом ~7 шагов/мин за 3 минуты. Ускорения в
  последней минуте нет, поэтому фильтр v2.10 «10 шагов + 1.5×» её не видел;
* `VTBR_SBER` 05.11 10:10 — вход на **незрелой σ** (`0.110 ₽` против `0.318 ₽` через час)
  после пересева EWMA на короткой истории (30 сэмплов).
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from bot_a.math_engine.pair_spread import PairSpreadCalculator
from bot_a.models.domain import ExecutionTopology, Side
from tests.helpers import make_book, make_meta
from tests.test_engine import _bundle, _params

SINGLE = ExecutionTopology.SINGLE_LEG_LEAD_LAG


# --------------------------------------------------------------- «медленный нож»
def _knife_engine(tmp_path, **overrides):
    base = {
        "use_anti_knife_filter": True,
        "anti_knife_min_steps": 10,
        "anti_knife_accel_multiple": 1.5,
        "anti_knife_cooldown_seconds": 600,
        "anti_knife_bleed_seconds": 180,
        "anti_knife_bleed_min_steps": 16,
        "anti_knife_bleed_sigma_multiple": 1.2,
        "anti_knife_bleed_max_lead_share": 0.15,
    }
    base.update(overrides)
    return _bundle(tmp_path, SINGLE, _params(**base))


def _fill_history(engine, now_ms: int, leg1: list, leg2: list) -> None:
    """Кладёт историю цен ног: ``leg1``/``leg2`` — по точке на 300/180/60/0 с назад."""
    engine._leg1_ticks.clear()  # noqa: SLF001
    engine._leg2_ticks.clear()  # noqa: SLF001
    for seconds, price in zip((300, 180, 60, 0), leg1):
        engine._leg1_ticks.append((now_ms - seconds * 1000, Decimal(price)))  # noqa: SLF001
    for seconds, price in zip((300, 180, 60, 0), leg2):
        engine._leg2_ticks.append((now_ms - seconds * 1000, Decimal(price)))  # noqa: SLF001


def _set_sigma(engine, sigma_rub: float, beta: float = 5.25) -> None:
    """Готовит `_last_metrics` с заданными σ спреда и бетой (σ_Leg1 = σ/β)."""
    from dataclasses import replace as dc_replace

    calc = engine._calc  # noqa: SLF001
    now = engine._clock.now_ms()  # noqa: SLF001
    calc.on_book(0, make_book(Decimal("100.00"), now, uid="uid-AAA"))
    metrics = calc.on_book(1, make_book(Decimal("500.00"), now, uid="uid-BBB"))
    assert metrics is not None
    engine._last_metrics = dc_replace(  # noqa: SLF001
        metrics, variance=sigma_rub**2, beta=beta
    )


def test_bleed_blocks_sustained_three_minute_bleed(tmp_path):
    """VTBR 02.11: 21 шаг = 1.25 σ за 3 мин при `lead_share = −0.24` — блок."""

    async def scenario():
        config, clock, _fee, gateway, engine = _knife_engine(tmp_path)
        await gateway.start()
        now = clock.now_ms()
        # 180 с назад: 100.21 → сейчас 100.00 = 21 шаг по 0.01 (кейс VTBR 02.11)
        _fill_history(
            engine,
            now,
            ["100.22", "100.21", "100.10", "100.00"],
            ["290.60", "290.58", "290.50", "290.42"],
        )
        _set_sigma(engine, 0.0845 * 5.25, beta=5.25)  # σ_Leg1 = 0.0845 ₽
        return engine._bleed_knife_reason(Side.BUY, now)  # noqa: SLF001

    reason = asyncio.run(scenario())
    assert reason is not None, "медленный нож обязан блокировать вход"
    assert "σ за 180 с" in reason
    assert "lead_share" in reason


def test_bleed_allows_genuine_lead_lag(tmp_path):
    """MTLR 28.10 (+14.23): обе ноги идут в сторону сделки, `lead_share = +0.18` — пропуск."""

    async def scenario():
        config, clock, _fee, gateway, engine = _knife_engine(tmp_path)
        await gateway.start()
        now = clock.now_ms()
        # SELL: Leg1 растёт против нас (0.38 ₽ за 3 мин), но Leg2 растёт сильнее по рублям.
        _fill_history(
            engine,
            now,
            ["100.00", "100.10", "100.20", "100.38"],
            ["290.00", "290.40", "290.70", "290.94"],
        )
        _set_sigma(engine, 0.098 * 6.45, beta=6.45)
        return engine._bleed_knife_reason(Side.SELL, now)  # noqa: SLF001

    assert asyncio.run(scenario()) is None


def test_bleed_allows_short_move(tmp_path):
    """IRAO 29.10 (+5.69): 13 шагов < порога 16 — фильтр не вмешивается."""

    async def scenario():
        config, clock, _fee, gateway, engine = _knife_engine(tmp_path)
        await gateway.start()
        now = clock.now_ms()
        _fill_history(
            engine, now, ["100.00", "100.05", "100.10", "100.10"], ["290.00", "290.00", "289.90", "289.90"]
        )
        _set_sigma(engine, 0.082 * 5.25, beta=5.25)
        return engine._bleed_knife_reason(Side.BUY, now)  # noqa: SLF001

    assert asyncio.run(scenario()) is None


def test_bleed_disabled_by_default(tmp_path):
    """Без `antiKnifeBleedMinSteps` поведение v2.11 (важно для прежних тестов)."""
    assert _params().anti_knife_bleed_min_steps == 0

    async def scenario():
        config, clock, _fee, gateway, engine = _knife_engine(tmp_path, anti_knife_bleed_min_steps=0)
        await gateway.start()
        now = clock.now_ms()
        _fill_history(
            engine,
            now,
            ["100.22", "100.21", "100.10", "100.00"],
            ["290.60", "290.58", "290.50", "290.42"],
        )
        _set_sigma(engine, 0.0845 * 5.25, beta=5.25)
        return engine._bleed_knife_reason(Side.BUY, now)  # noqa: SLF001

    assert asyncio.run(scenario()) is None


def test_bleed_arms_cooldown_for_the_same_side(tmp_path):
    """Сработавший медленный нож держит паузу и не блокирует обратную сторону."""

    async def scenario():
        config, clock, _fee, gateway, engine = _knife_engine(tmp_path)
        await gateway.start()
        now = clock.now_ms()
        _fill_history(
            engine,
            now,
            ["100.22", "100.21", "100.10", "100.00"],
            ["290.60", "290.58", "290.50", "290.42"],
        )
        _set_sigma(engine, 0.0845 * 5.25, beta=5.25)
        first = engine._anti_knife_reason(Side.BUY, now)  # noqa: SLF001
        assert first is not None
        pause = engine._anti_knife_reason(Side.BUY, now + 60_000)  # noqa: SLF001
        assert pause is not None and "пауза" in pause
        # Обратная сторона проверяется заново: цена уже не падает — фильтр молчит.
        _fill_history(
            engine,
            now + 60_000,
            ["100.00", "100.00", "100.00", "100.00"],
            ["290.10", "290.10", "290.10", "290.10"],
        )
        assert engine._anti_knife_reason(Side.SELL, now + 60_000) is None  # noqa: SLF001

    asyncio.run(scenario())


# --------------------------------------------------------------- зрелость σ
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
        min_samples_after_beta_reseed=50,
    )
    base.update(kwargs)
    return PairSpreadCalculator(**base)


def _feed(calc, time_ms: int, price1: str = "100.00", price2: str = "500.00"):
    calc.on_book(0, make_book(Decimal(price1), time_ms, uid="uid-AAA"))
    return calc.on_book(1, make_book(Decimal(price2), time_ms, uid="uid-BBB"))


def test_reseed_on_short_history_marks_stats_immature():
    """Короткое пересев (30 сэмплов) → σ незрелая, торговля запрещена."""
    calc = _calc()
    base = 1_700_000_000_000
    for index in range(30):
        _feed(calc, base + index * 5000)
    # на 30-м сэмпле бета оценена (beta_min_samples = 5) и история короткая
    assert calc.beta_reseeds >= 1
    assert calc.stats_mature is False, "после короткого пересева статистика незрелая"
    assert calc.post_reseed_samples < 50

    # Набираем 50 сэмплов после пересева → зрелость восстановлена.
    for index in range(30, 90):
        _feed(calc, base + index * 5000)
    assert calc.stats_mature is True, calc.post_reseed_samples
    assert calc.post_reseed_samples >= 50


def test_reseed_on_long_history_is_immediately_mature():
    """Длинная история: зрелость восстанавливается сразу после пересева (в прогреве)."""
    calc = _calc(min_samples_after_beta_reseed=10, ewma_window=40)
    base = 1_700_000_000_000
    for index in range(60):
        _feed(calc, base + index * 5000)
    assert calc.beta_reseeds >= 1
    assert calc.stats_mature is True, "за прогревочные сэмплы зрелость восстановилась"


def test_reseed_gate_disabled_by_default():
    """`minSamplesAfterBetaReseed = 0` — прежнее поведение (тесты v2.11 бит-в-бит)."""
    calc = _calc(min_samples_after_beta_reseed=0)
    base = 1_700_000_000_000
    for index in range(30):
        _feed(calc, base + index * 5000)
    assert calc.stats_mature is True


def test_immature_stats_block_entry(tmp_path):
    """Полный путь: незрелая σ режет входы и считает `reseedWarmupSkips`."""

    async def scenario():
        params = _params(
            beta_mode="slow_ratio",
            beta_min_samples=5,
            min_samples_after_beta_reseed=100_000,
        )
        config, clock, _fee, gateway, engine = _bundle(tmp_path, SINGLE, params)
        await gateway.start()
        from tests.test_engine import _tick, _warmup

        await _warmup(engine, clock, gateway)
        # После прогрева бета оценена на короткой истории → σ незрелая.
        assert engine._calc.beta_reseeds >= 1  # noqa: SLF001
        assert engine._calc.stats_mature is False  # noqa: SLF001
        for _ in range(2):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"))
        assert engine.position is None, "на незрелой σ входить нельзя"
        assert engine.counters["reseedWarmupSkips"] > 0

    asyncio.run(scenario())


def test_mature_stats_allow_entry(tmp_path):
    """Гейт выключен (0) — обычный вход работает как в v2.11."""

    async def scenario():
        params = _params(beta_mode="slow_ratio", beta_min_samples=5)
        config, clock, _fee, gateway, engine = _bundle(tmp_path, SINGLE, params)
        await gateway.start()
        from tests.test_engine import _open_position

        await _open_position(engine, clock, gateway)
        return engine

    engine = asyncio.run(scenario())
    assert engine.position is not None
