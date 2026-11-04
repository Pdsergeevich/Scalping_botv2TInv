"""Тесты v2.7: заморозка статистики на аукционах, потолок тейка, кластерный лимит.

Все четыре механики выросли из реального дня 02.11.2026 (`report_20261102.md`):

1. пре-маркет 09:20–10:00 отравлял EWMA и давал `|Z| = 12.98`;
2. `AFKS_SBER` (β = 0.377, `tpDeviationFraction = 4.0`) просил тейк +10 % хода цены;
3. три пары с общей ногой `SBER` вошли в одну сторону и дали −40.76 ₽;
4. стоп `VTBR` сработал за 20 с до возврата спреда, потому что σ выросла вдвое.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from bot_a.math_engine.pair_spread import (
    PairSpreadCalculator,
    msk_time_of_day_ms,
    parse_clock_msk,
    parse_clock_windows,
)
from bot_a.models.domain import ExecutionTopology
from bot_a.risk.cluster import ClusterRiskCoordinator, cluster_filter_trades
from tests.helpers import make_book, make_meta, trading_start_ms
from tests.test_engine import _bundle, _open_position, _params, _tick

DAY = 1_700_000_000_000 - (1_700_000_000_000 % 86_400_000)


def _msk(hour: int, minute: int, second: int = 0, base_day: int = DAY) -> int:
    return base_day + ((hour * 60 + minute) * 60 + second) * 1000 - 3 * 3600 * 1000


# --------------------------------------------------------------- парсинг настроек гейта
def test_parse_clock_helpers():
    assert parse_clock_msk("10:00:15") == ((10 * 60) * 60 + 15) * 1000
    assert parse_clock_msk("") is None
    assert parse_clock_msk("мусор") is None
    windows = parse_clock_windows("13:58-14:05,18:40-19:05")
    assert len(windows) == 2
    assert windows[0][0] < windows[0][1] < windows[1][0]
    assert parse_clock_windows("") == ()
    assert msk_time_of_day_ms(_msk(10, 0, 15)) == ((10 * 60) * 60 + 15) * 1000


# --------------------------------------------------------------- заморозка статистики
def _calc(**kwargs) -> PairSpreadCalculator:
    return PairSpreadCalculator(
        leg1=make_meta("AAA", uid="uid-AAA"),
        leg2=make_meta("BBB", uid="uid-BBB"),
        beta=1.0,
        ewma_window=20,
        beta_mode="slow_ratio",
        beta_min_samples=5,
        macro_beta_window=200,
        macro_beta_min_samples=10,
        stat_sample_interval_seconds=5.0,
        **kwargs,
    )


def _feed(calc, time_ms: int, price1: str = "100.00", price2: str = "101.00"):
    calc.on_book(0, make_book(Decimal(price1), time_ms, uid="uid-AAA"))
    return calc.on_book(1, make_book(Decimal(price2), time_ms, uid="uid-BBB"))


def test_premarket_ticks_do_not_move_statistics():
    """09:20–10:00 (аукцион) не двигает EWMA: |Z| = 12.98 больше не появится."""
    calc = _calc(stat_freeze_until_msk="10:00:15", stat_freeze_windows="")
    metrics = None
    for index in range(6):
        metrics = _feed(calc, _msk(9, 30 + index), "100.00", "104.00")
    assert metrics is not None
    assert metrics.stat_frozen is True
    assert "пре-маркет" in (metrics.freeze_reason or "")
    assert metrics.zscore == 0.0, "на аукционе Z не определён"
    assert calc.ewma.count == 0, "окно не должно наполняться пре-маркетом"
    assert calc.frozen_ticks >= 6

    # 10:00:20 — гейт открылся, статистика пошла
    metrics = _feed(calc, _msk(10, 0, 20), "100.00", "101.00")
    assert metrics.stat_frozen is False
    assert calc.ewma.count == 1


def test_clearing_auction_window_is_frozen():
    calc = _calc(stat_freeze_until_msk="10:00:15", stat_freeze_windows="13:58-14:05")
    before = _feed(calc, _msk(13, 57), "100.00", "101.00")
    inside = _feed(calc, _msk(13, 59), "100.00", "104.00")
    after = _feed(calc, _msk(14, 6), "100.00", "101.00")
    assert before.stat_frozen is False
    assert inside.stat_frozen is True
    assert "клиринг" in (inside.freeze_reason or "")
    assert after.stat_frozen is False


def test_wide_book_spread_is_frozen():
    """Спред стакана шире 50 bps — признак аукциона: статистику не двигаем."""
    calc = _calc(max_valid_spread_bps=50.0)
    # make_book(step="0.01") даёт ~1 bps — тик валиден
    assert _feed(calc, _msk(11, 0), "100.00", "101.00").stat_frozen is False
    # шаг 1.0 RUB при цене 100 = 100 bps — гейт должен отсечь
    wide = make_book(Decimal("100.00"), _msk(11, 1), uid="uid-AAA", step="1.0")
    calc.on_book(0, wide)
    metrics = calc.on_book(1, make_book(Decimal("101.00"), _msk(11, 1), uid="uid-BBB"))
    assert metrics.stat_frozen is True
    assert "широкий стакан" in (metrics.freeze_reason or "")


def test_gate_disabled_keeps_legacy_behaviour():
    """Пустые настройки — режим прежних версий: пре-маркет двигает статистику."""
    calc = _calc()
    metrics = _feed(calc, _msk(9, 30), "100.00", "101.00")
    assert metrics.stat_frozen is False
    assert calc.ewma.count == 1


def test_frozen_tick_blocks_entry_but_not_position_management(tmp_path):
    """Заморозка запрещает новые входы, но управление позицией продолжается."""
    from dataclasses import replace

    params = _params(stat_freeze_until_msk="10:00:15", stat_freeze_windows="")
    config, clock, _fee, gateway, engine = _bundle(
        tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
    )

    async def scenario() -> None:
        await gateway.start()
        # Прогрев после гейта: статистика наполняется, ядро выходит в Idle.
        for _ in range(80):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("101.00"))
        assert engine.state.name == "IDLE"
        metrics = engine.metrics
        assert metrics is not None
        frozen = replace(metrics, stat_frozen=True, freeze_reason="пре-маркет (тест)")
        before_entries = engine.counters["entries"]

        await engine._on_metrics(frozen)  # noqa: SLF001
        assert engine.counters["statFrozenSkips"] == 1
        assert engine.counters["entries"] == before_entries, "вход запрещён на аукционе"
        assert engine.position is None
        # С живой позицией управление не блокируется: стопы и выходы работают всегда.
        await _open_position(engine, clock, gateway)
        assert engine.state.name != "IDLE"
        await engine._on_metrics(replace(engine.metrics, stat_frozen=True))  # noqa: SLF001
        assert engine.counters["statFrozenSkips"] == 2
        assert engine.state.name != "IDLE", "позиция не должна закрываться из-за гейта"

    asyncio.run(scenario())


# --------------------------------------------------------------- потолок цели тейка
def test_tp_cap_keeps_target_realistic(tmp_path):
    """AFKS-кейс: цель ограничена 2.2σ и 0.45 % номинала, а не «+10 %»."""
    params = _params(tp_max_sigma_multiple=2.2, tp_max_notional_fraction=0.0045)
    config, clock, _fee, gateway, engine = _bundle(
        tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
    )

    async def scenario() -> None:
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        assert position is not None
        target = engine._target_tp_rub(position, clock.now_ms())  # noqa: SLF001
        price = engine._tp_price_for(position, target)  # noqa: SLF001
        entry = position.leg1_entry_price
        move = abs(price - entry) / entry
        assert move < Decimal("0.02"), f"тейк уехал на {move:.1%} от входа"
        assert target <= position.round_trip_fee_rub * Decimal("60")

    asyncio.run(scenario())


def test_tp_cap_disabled_keeps_old_target(tmp_path):
    """Без капов (нули) цель считается как раньше — обратная совместимость."""
    params = _params(tp_max_sigma_multiple=0.0, tp_max_notional_fraction=0.0)
    config, clock, _fee, gateway, engine = _bundle(
        tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
    )

    async def scenario() -> Decimal:
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        assert position is not None
        return engine._target_tp_rub(position, clock.now_ms())  # noqa: SLF001

    capped_target = asyncio.run(scenario())
    assert capped_target > Decimal("0")


# --------------------------------------------------------------- кластерный лимит
def test_cluster_blocks_second_same_direction():
    coordinator = ClusterRiskCoordinator(max_same_direction=1)
    assert coordinator.check("VTBR_SBER", "SBER", "BUY") is True
    coordinator.register("VTBR_SBER", "SBER", "BUY")
    # вторая пара с той же ногой и тем же направлением — блок
    assert coordinator.check("IRAO_SBER", "SBER", "BUY") is False
    assert coordinator.skips == 1
    # обратная сторона разрешена
    assert coordinator.check("IRAO_SBER", "SBER", "SELL") is True
    # другая нога — свой кластер
    assert coordinator.check("MTLR_GAZP", "GAZP", "BUY") is True
    coordinator.release("VTBR_SBER")
    assert coordinator.check("AFKS_SBER", "SBER", "BUY") is True


def test_cluster_limit_two_and_disabled():
    two = ClusterRiskCoordinator(max_same_direction=2)
    two.register("A", "SBER", "BUY")
    assert two.check("B", "SBER", "BUY") is True
    two.register("B", "SBER", "BUY")
    assert two.check("C", "SBER", "BUY") is False

    off = ClusterRiskCoordinator(max_same_direction=0)
    for name in ("A", "B", "C", "D"):
        assert off.check(name, "SBER", "BUY") is True
        off.register(name, "SBER", "BUY")


def test_cluster_filter_trades_removes_correlated_entries():
    """02.11: три `buy` в 11:22–11:36 по общей ноге — остаётся одна сделка."""
    trades = {
        "VTBR_SBER": [{"open_ms": 100, "close_ms": 900, "side": "BUY", "net": -19.26}],
        "IRAO_SBER": [{"open_ms": 200, "close_ms": 1900, "side": "BUY", "net": -20.18}],
        "AFKS_SBER": [{"open_ms": 800, "close_ms": 2600, "side": "BUY", "net": -1.33}],
    }
    legs = {"VTBR_SBER": "SBER", "IRAO_SBER": "SBER", "AFKS_SBER": "SBER"}

    kept, skipped = cluster_filter_trades(trades, legs, max_same_direction=1)

    assert skipped == 2
    assert len(kept["VTBR_SBER"]) == 1
    assert kept["IRAO_SBER"] == [] and kept["AFKS_SBER"] == []


def test_cluster_filter_allows_next_trade_after_close():
    trades = {
        "VTBR_SBER": [{"open_ms": 100, "close_ms": 200, "side": "BUY", "net": 1.0}],
        "IRAO_SBER": [{"open_ms": 300, "close_ms": 400, "side": "BUY", "net": 2.0}],
    }
    legs = {"VTBR_SBER": "SBER", "IRAO_SBER": "SBER"}

    kept, skipped = cluster_filter_trades(trades, legs, max_same_direction=1)

    assert skipped == 0
    assert len(kept["IRAO_SBER"]) == 1


def test_cluster_filter_unlimited_keeps_everything():
    trades = {
        "A": [{"open_ms": 1, "close_ms": 100, "side": "BUY"}],
        "B": [{"open_ms": 2, "close_ms": 100, "side": "BUY"}],
    }
    kept, skipped = cluster_filter_trades(trades, {"A": "SBER", "B": "SBER"}, max_same_direction=0)
    assert skipped == 0
    assert len(kept["A"]) == 1 and len(kept["B"]) == 1
