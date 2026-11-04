"""Тесты v2.10: анти-нож фильтр по импульсу Leg1 (`IRAO` 02.11 11:22:41 BUY).

Диагностика `scripts/lead_diagnostics.py` по шести сделкам на реальных L2 показала, что
убыточный вход отличался не направлением ведущей ноги (у всех пар Leg1 делает основную
работу — `lead_share` у прибыльных тоже ≈ 0), а **скоростью падения самой Leg1**:

* `IRAO` 11:22:41 BUY: 14.8 шага цены за последнюю минуту, ускорение 1.92 → `−15.91 RUB`;
* пять прибыльных сделок: 1.5–11.5 шагов, ускорение 0.82–1.15.

Фильтр блокирует вход, когда Leg1 идёт против сделки быстро И ускоряется, и держит паузу
`antiKnifeCooldownSeconds`, чтобы не «отпустить» вход ровно на дне (это давало `−14.31 RUB`).
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from bot_a.models.domain import ExecutionTopology, Side
from tests.test_engine import _bundle, _params

SINGLE = ExecutionTopology.SINGLE_LEG_LEAD_LAG
STEP = Decimal("0.01")


def _engine(tmp_path, **overrides):
    base = {
        "use_anti_knife_filter": True,
        "anti_knife_window_seconds": 60,
        "anti_knife_baseline_seconds": 300,
        "anti_knife_min_steps": 10,
        "anti_knife_accel_multiple": 1.5,
        "anti_knife_cooldown_seconds": 600,
    }
    base.update(overrides)
    return _bundle(tmp_path, SINGLE, _params(**base))


def _history(engine, now_ms: int, now: str, one_min: str, five_min: str) -> None:
    """Три точки истории Leg1: сейчас, минуту назад и пять минут назад."""
    engine._leg1_ticks.clear()  # noqa: SLF001
    engine._leg1_ticks.extend(  # noqa: SLF001
        [
            (now_ms - 300_000, Decimal(five_min)),
            (now_ms - 60_000, Decimal(one_min)),
            (now_ms, Decimal(now)),
        ]
    )


def test_knife_blocks_accelerating_adverse_move(tmp_path):
    """IRAO-кейс: 14 шагов против сделки за минуту при ускорении 1.84 — блок."""

    async def scenario():
        config, clock, _fee, gateway, engine = _engine(tmp_path)
        await gateway.start()
        now = clock.now_ms()
        # Цена падает: 100.38 → 100.14 → 100.00 (14 шагов за минуту, 38 за пять).
        _history(engine, now, "100.00", "100.14", "100.38")
        return engine._anti_knife_reason(Side.BUY, now)  # noqa: SLF001

    reason = asyncio.run(scenario())
    assert reason is not None
    assert "ускорении" in reason and "шагов" in reason


def test_knife_allows_decelerating_move(tmp_path):
    """Ралли в продажу, которое выдыхается (ускорение 0.5), — это вход по стратегии."""

    async def scenario():
        config, clock, _fee, gateway, engine = _engine(tmp_path)
        await gateway.start()
        now = clock.now_ms()
        # Цена растёт: 100.00 → 100.50 → 100.60 (10 шагов за минуту, 100 за пять).
        _history(engine, now, "100.60", "100.50", "100.00")
        return engine._anti_knife_reason(Side.SELL, now)  # noqa: SLF001

    assert asyncio.run(scenario()) is None


def test_knife_allows_small_move(tmp_path):
    """Мелкое движение (3 шага) не блокируется даже при большом ускорении."""
    async def scenario():
        config, clock, _fee, gateway, engine = _engine(tmp_path)
        await gateway.start()
        now = clock.now_ms()
        _history(engine, now, "100.00", "100.03", "100.035")
        return engine._anti_knife_reason(Side.BUY, now)  # noqa: SLF001

    assert asyncio.run(scenario()) is None


def test_knife_disabled_by_default(tmp_path):
    """Без `useAntiKnifeFilter` поведение прежних версий (важно для старых тестов)."""
    params = _params()
    assert params.use_anti_knife_filter is False

    async def scenario():
        config, clock, _fee, gateway, engine = _bundle(tmp_path, SINGLE, params)
        await gateway.start()
        now = clock.now_ms()
        engine._leg1_ticks.extend(  # noqa: SLF001
            [(now - 300_000, Decimal("100.38")), (now - 60_000, Decimal("100.14")),
             (now, Decimal("100.00"))]
        )
        return engine._anti_knife_reason(Side.BUY, now)  # noqa: SLF001

    assert asyncio.run(scenario()) is None


def test_knife_cooldown_blocks_only_the_same_side(tmp_path):
    """После срабатывания та же сторона ждёт паузу, обратная проверяется заново."""

    async def scenario():
        config, clock, _fee, gateway, engine = _engine(tmp_path)
        await gateway.start()
        now = clock.now_ms()
        _history(engine, now, "100.00", "100.14", "100.38")
        assert engine._anti_knife_reason(Side.BUY, now) is not None  # noqa: SLF001
        assert engine._knife_block_until[Side.BUY] == now + 600_000  # noqa: SLF001
        # Цена успокоилась, но пауза ещё идёт.
        _history(engine, now + 60_000, "100.00", "100.00", "100.14")
        pause = engine._anti_knife_reason(Side.BUY, now + 60_000)  # noqa: SLF001
        assert pause is not None and "пауза" in pause
        # Обратная сторона не заблокирована паузой и не видит ножа.
        assert engine._anti_knife_reason(Side.SELL, now + 60_000) is None  # noqa: SLF001

    asyncio.run(scenario())


def test_knife_blocks_entry_in_state_machine(tmp_path):
    """Полный путь: сигнал есть, но фильтр не даёт открыть позицию и считает отсечку."""

    async def flow():
        config, clock, _fee, gateway, engine = _engine(tmp_path)
        await gateway.start()
        # Прогреваем статистику без ножа, затем создаём «нож» и проверяем счётчик.
        from tests.test_engine import _tick, _warmup

        await _warmup(engine, clock, gateway)
        now = clock.now_ms()
        _history(engine, now, "100.00", "100.14", "100.38")
        engine._leg1_ticks[-1] = (now, Decimal("100.00"))  # noqa: SLF001
        before = engine.counters["antiKnifeSkips"]
        # Сигнал на покупку: спред ушёл вверх.
        for _ in range(2):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"))
            _history(engine, clock.now_ms(), "100.00", "100.14", "100.38")
            await engine._maybe_enter(engine.metrics)  # noqa: SLF001
            if engine.position is not None:
                break
        assert engine.position is None, "анти-нож не должен пускать в падающий слейв"
        assert engine.counters["antiKnifeSkips"] > before

    asyncio.run(flow())
