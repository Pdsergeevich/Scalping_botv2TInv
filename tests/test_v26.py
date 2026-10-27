"""Тесты v2.6: персональные настройки пары (окно EWMA и порог σ/Fee)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

from bot_a.broker.tinkoff_rest_client import TInvestRestError
from bot_a.config import AppConfig, PairPreset
from bot_a.models.domain import ExecutionTopology
from bot_a.services.multi_preset import THROTTLE_MIN_SECONDS
from tests.helpers import make_config, make_meta, make_preset
from tests.test_engine import _bundle, _params, _tick, _warmup


def _preset_with(**overrides) -> PairPreset:
    leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
    leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
    base = replace(make_preset(leg1, leg2), name="AAA_BBB")
    return replace(base, **overrides)


def test_preset_defaults_have_no_overrides():
    preset = _preset_with()
    assert preset.ewma_window is None
    assert preset.min_sigma_to_fee_ratio is None


def test_with_preset_overrides_applies_ewma_and_sigma(tmp_path):
    """v2.6: пара может задать своё окно EWMA и свой порог σ/Fee."""
    preset = _preset_with(ewma_window=1200, min_sigma_to_fee_ratio=1.05, z_entry=2.6)
    config = make_config(tmp_path, preset, _params())
    config = replace(config, presets={"AAA_BBB": preset}, active_preset="AAA_BBB")

    updated = config.with_preset_overrides()

    assert updated.strategy.ewma_window == 1200
    assert updated.strategy.min_sigma_to_fee_ratio == 1.05
    assert updated.strategy.z_entry == 2.6


def test_preset_overrides_keep_global_values_when_absent(tmp_path):
    """Если пара ничего не задала, глобальные значения стратегии не меняются."""
    preset = _preset_with()
    config = make_config(tmp_path, preset, _params())
    config = replace(config, presets={"AAA_BBB": preset}, active_preset="AAA_BBB")
    before = (config.strategy.ewma_window, config.strategy.min_sigma_to_fee_ratio)

    updated = config.with_preset_overrides()

    assert (updated.strategy.ewma_window, updated.strategy.min_sigma_to_fee_ratio) == before


def test_preset_overrides_parsed_from_json(tmp_path):
    """Новые ключи читаются из appsettings.json (`ewmaWindow`, `minSigmaToFeeRatio`)."""
    import json

    raw = {
        "presets": {
            "X_Y": {
                "description": "тест",
                "leg1": {"ticker": "X", "classCode": "TQBR", "lotSize": 10},
                "leg2": {"ticker": "Y", "classCode": "TQBR", "lotSize": 1},
                "enabled": True,
                "zEntry": 2.8,
                "ewmaWindow": 900,
                "minSigmaToFeeRatio": 0.95,
            }
        }
    }
    path = tmp_path / "appsettings.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    config = AppConfig.load(path)
    preset = config.preset("X_Y")

    assert preset.ewma_window == 900
    assert preset.min_sigma_to_fee_ratio == 0.95
    applied = config.with_preset_overrides()
    assert applied.strategy.ewma_window == 900
    assert applied.strategy.min_sigma_to_fee_ratio == 0.95


def test_multi_preset_runner_uses_per_pair_ewma(tmp_path):
    """ALL_ACTIVE собирает параметры каждой пары с её собственным окном EWMA."""
    from bot_a.services.multi_preset import MultiPresetRunner

    fast = _preset_with(name="FAST_PAIR", ewma_window=900, min_sigma_to_fee_ratio=1.05)
    slow = _preset_with(name="SLOW_PAIR")
    config = make_config(tmp_path, fast, _params())
    config = replace(config, presets={"FAST_PAIR": fast, "SLOW_PAIR": slow})

    runner = MultiPresetRunner(
        config, [fast, slow], client=None, interval_ms=100, serve_dashboard=False
    )

    params_fast = runner._params_for(fast)  # noqa: SLF001
    params_slow = runner._params_for(slow)  # noqa: SLF001

    assert params_fast.ewma_window == 900
    assert params_fast.min_sigma_to_fee_ratio == 1.05
    assert params_slow.ewma_window == config.strategy.ewma_window
    assert params_slow.min_sigma_to_fee_ratio == config.strategy.min_sigma_to_fee_ratio


# --------------------------------------------- v2.6.2: форма тейка и размер позиции
def test_preset_tp_overrides_applied(tmp_path):
    """Пара может задать свою форму тейка: tpDeviationFraction и tpMinFeeMultiple."""
    preset = _preset_with(
        name="TP_PAIR", tp_deviation_fraction=4.0, tp_min_fee_multiple=3.5, quantity_lots=2
    )
    config = make_config(tmp_path, preset, _params())
    config = replace(config, presets={"TP_PAIR": preset}, active_preset="TP_PAIR")

    applied = config.with_preset_overrides()

    assert applied.strategy.tp_deviation_fraction == 4.0
    assert applied.strategy.tp_min_fee_multiple == Decimal("3.5")
    assert applied.strategy.quantity_lots == 2


def test_preset_tp_overrides_keep_global_when_absent(tmp_path):
    preset = _preset_with(name="PLAIN_PAIR")
    config = make_config(tmp_path, preset, _params())
    config = replace(config, presets={"PLAIN_PAIR": preset}, active_preset="PLAIN_PAIR")
    before = (config.strategy.tp_deviation_fraction, config.strategy.tp_min_fee_multiple)

    applied = config.with_preset_overrides()

    assert (applied.strategy.tp_deviation_fraction, applied.strategy.tp_min_fee_multiple) == before


def test_multi_preset_runner_applies_tp_and_lots(tmp_path):
    """ALL_ACTIVE применяет форму тейка и число лотов каждой пары отдельно."""
    from bot_a.services.multi_preset import MultiPresetRunner

    wide = _preset_with(name="WIDE", tp_deviation_fraction=4.0, quantity_lots=2)
    plain = _preset_with(name="PLAIN")
    config = make_config(tmp_path, wide, _params())
    config = replace(config, presets={"WIDE": wide, "PLAIN": plain})

    runner = MultiPresetRunner(
        config, [wide, plain], client=None, interval_ms=100, serve_dashboard=False
    )
    params_wide = runner._params_for(wide)  # noqa: SLF001
    params_plain = runner._params_for(plain)  # noqa: SLF001

    assert params_wide.tp_deviation_fraction == 4.0
    assert params_wide.quantity_lots == 2
    assert params_plain.tp_deviation_fraction == config.strategy.tp_deviation_fraction
    assert params_plain.quantity_lots == config.strategy.quantity_lots


# ------------------------------- v2.6.3: фильтр σ/Fee должен быть инвариантен к лотам
def _vol_filter_ratio(lots: int, tmp_path) -> tuple[float, int]:
    """Возвращает (σ_leg1 позиции / комиссия позиции, счётчик отсечек vol-фильтра)."""

    async def run() -> tuple[float, int]:
        params = _params(
            z_entry=0.5,
            quantity_lots=lots,
            use_vol_to_fee_filter=True,
            min_sigma_to_fee_ratio=1.30,
        )
        _config, clock, _fee, gateway, engine = _bundle(
            tmp_path / f"lots{lots}", ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
        )
        # Тестовый конфиг держит риск-потолок в 2 лота — поднимаем его, иначе
        # предохранитель урежет позицию и сравнение лотов станет бессмысленным.
        engine._guard._max_lots = lots  # noqa: SLF001
        await gateway.start()
        await _warmup(engine, clock, gateway)
        metrics = engine._last_metrics  # noqa: SLF001
        sigma_position = engine._sigma_leg1_rub(metrics) * lots  # noqa: SLF001
        fee_position = engine._current_round_trip_fee(lots)  # noqa: SLF001
        assert fee_position > 0
        ratio = float(sigma_position / fee_position)
        # Провоцируем сигнал на вход и смотрим, отсекает ли его vol-фильтр.
        for _ in range(2):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"))
        return ratio, int(engine._counters["volFilterSkips"])  # noqa: SLF001

    return asyncio.run(run())


def test_sigma_to_fee_ratio_is_lot_invariant(tmp_path):
    """σ_leg1 позиции и комиссия позиции масштабируются одинаково.

    Регресс 27.10.2026: при `quantityLots = 55` комиссия умножалась на 55, а σ
    оставалась на один лот — отношение падало в 55 раз (0.09 вместо 4.9), и
    vol-фильтр блокировал любые входы в живом портфеле.
    """
    ratio_single, skips_single = _vol_filter_ratio(1, tmp_path)
    ratio_many, skips_many = _vol_filter_ratio(55, tmp_path)

    assert abs(ratio_single - ratio_many) / ratio_single < 0.01, (ratio_single, ratio_many)
    # Просадка отношения ровно в число лотов — это и был баг.
    assert ratio_many * 55 > ratio_single * 10
    # Поведение фильтра одинаково независимо от размера позиции.
    assert skips_single == skips_many


def test_risk_guard_clamps_preset_lots(tmp_path):
    """`risk.maxPositionLots` остаётся настоящим потолком для персональных лотов пары.

    При `maxPositionLots = 2` пара с `quantityLots = 55` входит двумя лотами —
    и это видно в логе предупреждением о урезании.
    """
    params = _params(z_entry=0.5, quantity_lots=55)
    config, clock, _fee, gateway, engine = _bundle(
        tmp_path, ExecutionTopology.SINGLE_LEG_LEAD_LAG, params
    )
    assert config.risk.max_position_lots == 2, "ожидается дефолт риск-настроек теста"
    assert engine._guard.clamp_quantity(55) == 2  # noqa: SLF001

    async def run() -> None:
        await gateway.start()
        await _warmup(engine, clock, gateway)
        for _ in range(2):
            await _tick(engine, clock, gateway, Decimal("100.00"), Decimal("104.00"))

    asyncio.run(run())
    assert engine.position is not None
    assert engine.position.quantity == 2, "позиция урезана предохранителем"


# ------------------------------------- v2.6.3: backoff на HTTP 429 (rate limit REST)
class _RateLimitedClient:
    async def get_order_book(self, uid: str, depth: int = 20):
        raise TInvestRestError("HTTP 429", status=429, body="")


class _BrokenClient:
    async def get_order_book(self, uid: str, depth: int = 20):
        raise TInvestRestError("HTTP 500", status=500, body="")


def _runtime_stub(preset) -> SimpleNamespace:
    syncer = SimpleNamespace(
        skew_skips=0,
        check=lambda fresh, fetched, failed: SimpleNamespace(ok=False, reason="нога не ответила"),
    )
    engine = SimpleNamespace(on_clock=_noop)
    return SimpleNamespace(
        preset=preset,
        engine=engine,
        syncer=syncer,
        books={},
        rest_errors=0,
        rate_limited=0,
        snapshots=0,
        name=preset.name,
    )


async def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def _runner(tmp_path, preset, client) -> Any:
    from bot_a.services.multi_preset import MultiPresetRunner

    config = make_config(tmp_path, preset, _params())
    return MultiPresetRunner(
        config, [preset], client=client, interval_ms=100, serve_dashboard=False
    )


def test_rate_limit_triggers_throttle(tmp_path):
    """429 не считается обычной ошибкой: пара уходит в паузу, а темп опроса падает."""
    preset = _preset_with(name="RATE")
    runner = _runner(tmp_path, preset, _RateLimitedClient())
    runtime = _runtime_stub(preset)

    before = runner._throttle_seconds  # noqa: SLF001
    asyncio.run(runner._poll_pair(runtime))  # noqa: SLF001

    assert runtime.rate_limited == 2, "по одному 429 на каждую ногу"
    assert runtime.rest_errors == 2
    assert runner._throttle_seconds > before  # noqa: SLF001
    assert runner._throttle_until > time.monotonic()  # noqa: SLF001


def test_other_errors_do_not_throttle(tmp_path):
    preset = _preset_with(name="PLAIN")
    runner = _runner(tmp_path, preset, _BrokenClient())
    runtime = _runtime_stub(preset)

    asyncio.run(runner._poll_pair(runtime))  # noqa: SLF001

    assert runtime.rest_errors == 2
    assert runtime.rate_limited == 0
    assert runner._throttle_until == 0.0  # noqa: SLF001


def test_throttle_decays_after_quiet_period(tmp_path):
    preset = _preset_with(name="QUIET")
    runner = _runner(tmp_path, preset, _RateLimitedClient())
    runtime = _runtime_stub(preset)
    asyncio.run(runner._poll_pair(runtime))  # noqa: SLF001
    grown = runner._throttle_seconds  # noqa: SLF001

    runner._throttle_until = 0.0  # noqa: SLF001, SLF002
    runner._last_rate_limit = time.monotonic() - 120.0  # noqa: SLF001
    asyncio.run(runner._wait_throttle())  # noqa: SLF001

    assert runner._throttle_seconds < grown  # noqa: SLF001
    assert runner._throttle_seconds >= THROTTLE_MIN_SECONDS  # noqa: SLF001
