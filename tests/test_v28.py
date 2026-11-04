"""Тесты v2.8: стоп от σ больше не сжимается потолком тейка.

Причина патча — `AFKS_SBER` на реальных L2 02.11.2026: потолок тейка
(`tpMaxSigmaMultiple` / `tpMaxNotionalFraction`) урезал цель до 4.80 ₽, а стоп
считался как `singleLegStopToTargetRatio × цель` и сжался с 248 ₽ до 10.56 ₽ →
позицию выбило шумом за 60 секунд. Дополнительно: тайм-аут удержания в плюсе
закрывался рынком как `HardStop` и попадал в дневной предохранитель.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from decimal import Decimal

from bot_a.math_engine.pair_spread import PairSpreadCalculator
from bot_a.models.domain import ExecutionTopology, ExitReason
from tests.helpers import make_book, make_meta
from tests.test_engine import _bundle, _open_position, _params, _tick
from tests.test_v27 import _msk

SINGLE = ExecutionTopology.SINGLE_LEG_LEAD_LAG


def _stop_inputs(engine, position):
    """σ Leg1 позиции и неурезанная реверсия — те же формулы, что в `fsm`."""
    beta = position.frozen_beta if position.frozen_beta > 1e-9 else 1.0
    lots = max(position.leg1_filled_qty, 1)
    sigma_pos = Decimal(str(position.frozen_sigma / beta * lots))
    raw_dev = abs(position.entry_spread - position.frozen_mu) / beta * lots
    return sigma_pos, Decimal(str(raw_dev))


def test_tp_cap_does_not_shrink_stop(tmp_path):
    """Потолок тейка режет цель, но НЕ трогает порог стопа (главный баг v2.7)."""

    async def scenario():
        capped = _params(
            tp_max_sigma_multiple=0.05,
            tp_max_notional_fraction=0.00001,
            stop_sigma_multiple=3.5,
        )
        config, clock, _fee, gateway, engine = _bundle(tmp_path / "cap", SINGLE, capped)
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        assert position is not None
        engine._freeze_position(position, engine.metrics)  # noqa: SLF001
        target = engine._target_tp_rub(position, clock.now_ms())  # noqa: SLF001
        stop_rub, source = engine._stop_threshold_rub(position, engine.metrics)  # noqa: SLF001
        sigma_pos, raw_dev = _stop_inputs(engine, position)

        legacy = target * Decimal(str(capped.single_leg_stop_to_target_ratio))
        assert target > 0
        assert stop_rub > legacy * Decimal("10"), (
            f"стоп всё ещё привязан к урезанной цели: {stop_rub} <= {legacy}×10"
        )
        assert stop_rub >= Decimal("3.5") * sigma_pos
        assert stop_rub >= Decimal(str(capped.single_leg_stop_to_target_ratio)) * raw_dev
        assert "σ_leg1" in source or "реверсия" in source
        return stop_rub, sigma_pos, target

    stop_rub, sigma_pos, target = asyncio.run(scenario())
    assert stop_rub > sigma_pos, "стоп должен быть шире одной σ"
    assert stop_rub > target, "стоп шире цели — это норма для скальпинга"


def test_stop_threshold_has_sigma_floor(tmp_path):
    """`StopThresholdRub >= stopSigmaMultiple × σ_Leg1,позиции` при любых капах."""

    async def scenario(k_sigma: float):
        params = _params(
            tp_max_sigma_multiple=2.2,
            tp_max_notional_fraction=0.0045,
            stop_sigma_multiple=k_sigma,
        )
        config, clock, _fee, gateway, engine = _bundle(
            tmp_path / f"k{k_sigma}", SINGLE, params
        )
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        assert position is not None
        stop_rub, _source = engine._stop_threshold_rub(position, engine.metrics)  # noqa: SLF001
        sigma_pos, _raw = _stop_inputs(engine, position)
        return stop_rub, sigma_pos

    for k_sigma in (3.0, 3.5, 4.0):
        stop_rub, sigma_pos = asyncio.run(scenario(k_sigma))
        assert stop_rub >= Decimal(str(k_sigma)) * sigma_pos


def test_stop_sigma_disabled_keeps_legacy_behaviour(tmp_path):
    """`stopSigmaMultiple = 0` — прежняя формула (цель × ratio), обратная совместимость."""

    async def scenario():
        params = _params(stop_sigma_multiple=0.0)
        config, clock, _fee, gateway, engine = _bundle(tmp_path, SINGLE, params)
        await gateway.start()
        await _open_position(engine, clock, gateway)
        position = engine.position
        assert position is not None
        stop_rub, source = engine._stop_threshold_rub(position, engine.metrics)  # noqa: SLF001
        target = engine._target_tp_rub(position, clock.now_ms())  # noqa: SLF001
        return stop_rub, target, source

    stop_rub, target, source = asyncio.run(scenario())
    assert "цель" in source
    assert stop_rub >= target * Decimal(str(_params().single_leg_stop_to_target_ratio)) - Decimal("1e-9")


def test_timeout_in_profit_exits_passively_not_hard_stop(tmp_path):
    """VTBR-кейс: 30-минутный выход в плюсе — пассивный TimeStop, не HardStop."""

    async def scenario():
        params = _params(max_hold_seconds=5, max_exit_quote_seconds=5)
        config, clock, _fee, gateway, engine = _bundle(tmp_path, SINGLE, params)
        await gateway.start()
        await _open_position(engine, clock, gateway)
        # Дальше Z «уходит против позиции» (как в 02.11): при старом коде тайм-аут
        # закрывался рынком как HardStop и попадал в дневной предохранитель.
        engine._params = replace(params, z_stop=0.5)  # noqa: SLF001
        for _ in range(6):
            clock.advance(1000)
            await engine.on_clock()
        assert engine.ledger.trade_count == 0, "выход должен быть пассивным (лимит в стакане)"
        assert engine.state.name == "QUOTING_EXIT_LEG1", engine.state
        assert engine.daily_hard_stops == 0, "тайм-аут не должен считаться HardStop"

    asyncio.run(scenario())


def test_timeout_beyond_stop_still_flattens_hard(tmp_path):
    """Если к тайм-ауту убыток уже за порогом стопа — закрываем рынком."""

    async def scenario():
        params = _params(
            max_hold_seconds=5,
            max_exit_quote_seconds=1,
            z_stop=100.0,
            stop_sigma_multiple=0.01,
            single_leg_stop_to_target_ratio=0.01,
            single_leg_stop_min_steps=1,
            use_single_leg_price_stop=True,
            lead_failure_z=100.0,
        )
        config, clock, _fee, gateway, engine = _bundle(tmp_path, SINGLE, params)
        await gateway.start()
        await _open_position(engine, clock, gateway)
        for _ in range(12):
            await _tick(engine, clock, gateway, Decimal("99.00"), Decimal("104.00"))
            if engine.ledger.trade_count:
                break
        assert engine.ledger.trade_count == 1, f"состояние={engine.state}"
        return engine.ledger.trades[0]

    trade = asyncio.run(scenario())
    assert trade.exit_reason is ExitReason.HARD_STOP
    assert trade.net_pnl_rub < 0


def test_bootstrap_bypasses_premarket_gate():
    """Прогрев историей идёт мимо гейта: иначе запуск до 10:00 MSK оставлял базу пустой."""
    calc = PairSpreadCalculator(
        leg1=make_meta("AAA", uid="uid-AAA"),
        leg2=make_meta("BBB", uid="uid-BBB"),
        beta=1.0,
        ewma_window=20,
        beta_min_samples=5,
        macro_beta_window=200,
        macro_beta_min_samples=10,
        stat_sample_interval_seconds=0.0,
        stat_freeze_until_msk="10:00:15",
    )
    calc.on_book(0, make_book(Decimal("100.00"), _msk(9, 30), uid="uid-AAA"))
    metrics = calc.on_book(1, make_book(Decimal("101.00"), _msk(9, 30), uid="uid-BBB"))
    assert metrics.stat_frozen is True, "боевой поток до 10:00:15 заморожен"
    assert calc.ewma.count == 0

    calc.set_stat_gate_bypass(True)
    for index in range(6):
        calc.on_book(0, make_book(Decimal("100.00"), _msk(9, 31 + index), uid="uid-AAA"))
        metrics = calc.on_book(
            1, make_book(Decimal("101.00"), _msk(9, 31 + index), uid="uid-BBB")
        )
    assert metrics.stat_frozen is False
    assert calc.ewma.count > 0, "прогрев обязан наполнять статистику"


def test_truncated_recording_reads_all_members(tmp_path):
    """Оборванная запись (испорченный хвост + новый gzip-член) читается целиком."""
    import gzip

    from bot_a.backtest.tick_replay import JsonlRecordingSource

    path = tmp_path / "PAIR_2026-11-02.jsonl.gz"
    first = b"".join(_book_row(index) for index in range(3))
    second = b"".join(_book_row(index) for index in range(3, 6))
    with open(path, "wb") as raw:
        raw.write(gzip.compress(first)[:-8])  # убитый процесс: хвост члена испорчен
        raw.write(gzip.compress(second))

    events = list(JsonlRecordingSource([path])._iter_file(path))  # noqa: SLF001

    assert len(events) == 6, f"прочитано {len(events)} из 6 строк"
    assert events[-1].time_ms > events[0].time_ms


def _book_row(index: int) -> bytes:
    payload = {
        "t": 1_700_000_000_000 + index * 1000,
        "k": "book",
        "leg": 1,
        "uid": "uid-AAA",
        "b": [["100.00", 10]],
        "a": [["100.02", 10]],
        "lp": "100.01",
    }
    return (json.dumps(payload) + "\n").encode("utf-8")
