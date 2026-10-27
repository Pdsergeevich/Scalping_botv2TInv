"""Тесты v2.6: персональные настройки пары (окно EWMA и порог σ/Fee)."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal

from bot_a.config import AppConfig, PairPreset
from bot_a.models.domain import ExecutionTopology
from tests.helpers import make_config, make_meta, make_preset
from tests.test_engine import _params


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
    a