"""Проверки загрузки appsettings.json и переопределений CLI."""

from decimal import Decimal
from pathlib import Path

import pytest

from bot_a.config import AppConfig
from bot_a.economics.fees import Tariff
from bot_a.engine.fsm import StrategyParams
from bot_a.models.domain import ExecutionTopology

ROOT = Path(__file__).resolve().parents[1]


def test_load_repository_config():
    config = AppConfig.load(ROOT / "appsettings.json")
    assert set(config.preset_names()) == {
        "AFKS_SBER",
        "CNYRUBF_CRZ6",
        "IMOEXF_MMZ6",
        "IRAO_SBER",
        "MTLR_SBER",
        "SBERP_SBER",
        "SPZ6_SRZ6",
        "SVZ6_GDZ6",
        "T_SBER",
        "VTBR_SBER",
    }
    preset = config.preset()
    assert preset.name == "VTBR_SBER"
    assert preset.enabled is True
    cy = config.preset("CNYRUBF_CRZ6")
    assert cy.enabled is False
    assert cy.leg1.is_futures and cy.leg1.class_code == "SPBFUT"
    assert cy.leg1.rub_per_price_unit == Decimal("1000")
    assert cy.leg1.position_margin_rub(1) == Decimal("983.48")
    assert set(config.trading_preset_names()) == {
        "IRAO_SBER",
        "MTLR_SBER",
        "VTBR_SBER",
    }
    # v2.8: AFKS_SBER выключена — на реальных L2-записях она убыточна.
    assert config.preset("AFKS_SBER").enabled is False
    assert config.disabled_preset_names() == [
        "AFKS_SBER",
        "CNYRUBF_CRZ6",
        "IMOEXF_MMZ6",
        "SBERP_SBER",
        "SPZ6_SRZ6",
        "SVZ6_GDZ6",
        "T_SBER",
    ]
    # v2.7: тариф по умолчанию Premium — live/watch не должны случайно
    # работать на Trader-комиссиях.
    assert config.fees.tariff is Tariff.PREMIUM
    assert config.execution.topology is ExecutionTopology.SINGLE_LEG_LEAD_LAG
    assert config.strategy.z_entry == 2.8
    assert config.strategy.ewma_window == 2400
    assert config.strategy.max_hold_seconds == 1800
    assert config.strategy.z_stop == 6.0
    assert config.strategy.use_frozen_mu is True
    assert config.strategy.use_profit_lock is True
    # v2.5: "auto" в appsettings резолвится в slow_ratio (раньше молча падал в
    # same_underlying внутри калькулятора, и бэктест считал бету иначе, чем live).
    assert config.strategy.beta_mode == "slow_ratio"
    assert preset.beta_mode == "slow_ratio"
    assert cy.beta_mode == "same_underlying"
    assert config.preset("T_SBER").beta_mode == "slow_ratio"
    assert config.strategy.min_net_profit_multiplier == Decimal("1.35")
    assert config.strategy.tp_deviation_fraction == 0.85
    # v2.7: заморозка статистики на аукционах и потолок цели тейка.
    assert config.strategy.stat_freeze_until_msk == "10:00:15"
    assert config.strategy.max_valid_spread_bps == 50.0
    # v2.9: потолок тейка поднят и включается только для пар с β < 0.8
    # (AFKS требовал +10 %; VTBR/MTLR теряли половину прибыли).
    assert config.strategy.tp_max_notional_fraction == 0.015
    assert config.strategy.tp_max_sigma_multiple == 4.5
    assert config.strategy.tp_cap_beta_threshold == 0.8
    assert config.strategy.use_volatility_adaptive_stop is True
    # v2.9: два однонаправленных входа на общую ногу вместо одного.
    assert config.risk.max_concurrent_same_direction == 2
    # v2.9: выход из «мёртвого» спреда через 13 минут.
    assert config.strategy.use_stagnation_exit is True
    assert config.strategy.stagnation_after_seconds == 780
    # v2.10: анти-нож по импульсу Leg1 (IRAO 02.11 11:22:41 BUY).
    assert config.strategy.use_anti_knife_filter is True
    assert config.strategy.anti_knife_min_steps == 10
    assert config.strategy.anti_knife_accel_multiple == 1.5
    assert config.strategy.anti_knife_cooldown_seconds == 600
    assert config.strategy.time_stop_quote_seconds == 30
    # v2.2: хук пика Z, сжатие тейка, режимный фильтр
    assert config.strategy.use_z_peak_hook is True
    assert config.strategy.z_hook_pullback == 0.12
    assert config.strategy.z_min_trigger == 0.15
    assert config.strategy.z_peak_reset_gap == 0.25
    # v2.3: выпуклость (ratchet и сжатие выключены), ценовой стоп Leg1, окна входа
    assert config.strategy.use_tp_decay is False
    assert config.strategy.use_breakeven_ratchet is False
    assert config.strategy.tp_min_fee_multiple == Decimal("2.5")
    assert config.strategy.use_single_leg_price_stop is True
    # v2.7: стоп шире цели (2.2 вместо 1.5) + 45 с подтверждения — иначе рост σ
    # выбивает позицию за секунды до возврата спреда (кейс VTBR 02.11).
    assert config.strategy.single_leg_stop_to_target_ratio == 2.2
    assert config.strategy.single_leg_stop_min_steps == 4
    assert config.strategy.lead_failure_z == 0.5
    assert config.strategy.lead_failure_seconds == 60
    assert config.risk.active_session_windows == "PrimeHoursOnly"
    # v2.4: дневной предохранитель, макро-дрейф беты и фильтр σ/Fee
    assert config.strategy.use_daily_stop_breaker is True
    assert config.strategy.max_daily_hard_stops == 1
    assert config.strategy.max_daily_loss_fee_multiple == Decimal("3.5")
    assert config.strategy.use_macro_beta_drift_guard is True
    assert config.strategy.macro_beta_window == 14400
    assert config.strategy.max_beta_drift == 0.008
    assert config.strategy.use_vol_to_fee_filter is True
    assert config.strategy.min_sigma_to_fee_ratio == 1.30
    assert config.strategy.ratchet_arm_multiplier == Decimal("2.2")
    assert config.strategy.ratchet_arm_target_fraction == 0.90
    assert config.strategy.ratchet_exit_multiplier == Decimal("1.15")
    assert config.strategy.ratchet_exit_target_fraction == 0.60
    assert not hasattr(config.strategy, "ratchet_z_exit")
    assert config.strategy.tp_decay_start_fraction == 0.5
    assert config.strategy.tp_decay_floor_multiplier == Decimal("1.15")
    assert config.strategy.use_regime_filter is True
    assert config.strategy.regime_window == 12
    assert config.strategy.regime_dominant_share == 0.65
    assert config.strategy.regime_stable_share == 0.35
    assert config.execution.queue_fill_factor == 0.35
    assert config.risk.margin_safety_buffer_pct == 25.0


def test_spot_preset_has_no_margin():
    config = AppConfig.load(ROOT / "appsettings.json")
    preset = config.preset("SBERP_SBER")
    assert not preset.leg1.is_futures
    assert preset.leg1.lot_size == 1
    assert preset.leg1.position_margin_rub(1) == Decimal("0")
    assert preset.leg1.rub_per_price_unit == Decimal("1")


def test_overrides_do_not_mutate_source():
    config = AppConfig.load(ROOT / "appsettings.json")
    updated = config.with_overrides(
        tariff=Tariff.PREMIUM,
        topology=ExecutionTopology.DUAL_LEG_HEDGED,
        strategy=StrategyParams(z_entry=2.5, z_exit=0.4, ewma_window=600),
        active_preset="SPZ6_SRZ6",
        max_daily_drawdown_rub=Decimal("5000"),
    )
    # v2.7: тариф по умолчанию Premium — live/watch не должны случайно
    # работать на Trader-комиссиях.
    assert config.fees.tariff is Tariff.PREMIUM
    assert updated.fees.tariff is Tariff.PREMIUM
    assert updated.execution.topology is ExecutionTopology.DUAL_LEG_HEDGED
    assert updated.strategy.z_entry == 2.5
    assert updated.preset().name == "SPZ6_SRZ6"
    assert updated.risk.max_daily_drawdown_rub == Decimal("5000")


def test_preset_overrides_lock_champion_configuration():
    """v2.3.1: порог входа, топология и окна входа задаются на уровне пресета."""
    config = AppConfig.load(ROOT / "appsettings.json")
    vtbr = config.with_preset_overrides()
    assert vtbr.preset().name == "VTBR_SBER"
    assert vtbr.strategy.z_entry == 2.8
    assert vtbr.execution.topology is ExecutionTopology.SINGLE_LEG_LEAD_LAG
    assert vtbr.risk.active_session_windows == "PrimeHoursOnly"

    sberp = config.with_preset_overrides("SBERP_SBER")
    assert sberp.strategy.z_entry == 3.2
    assert sberp.risk.active_session_windows == "PrimeHoursOnly"

    # исходная конфигурация не мутируется
    assert config.strategy.z_entry == 2.8
    assert config.preset("VTBR_SBER").z_entry == 2.8

    # пресет без переопределений наследует глобальные настройки
    plain = config.with_preset_overrides("SVZ6_GDZ6")
    assert plain.strategy.z_entry == config.strategy.z_entry
    assert plain.risk.active_session_windows == config.risk.active_session_windows


def test_disabled_presets_block_trading():
    config = AppConfig.load(ROOT / "appsettings.json")
    for name in ("T_SBER", "CNYRUBF_CRZ6", "SBERP_SBER"):
        assert config.preset(name).enabled is False
    preset = config.preset("VTBR_SBER")
    assert preset.enabled is True
    assert preset.z_entry == 2.8
    assert preset.topology == "SingleLegLeadLag"
    assert preset.active_session_windows == "PrimeHoursOnly"
    # SBERP переопределения остаются в конфиге (для paper/исследований), но торговля выключена.
    sberp = config.preset("SBERP_SBER")
    assert sberp.z_entry == 3.2
    assert sberp.active_session_windows == "PrimeHoursOnly"


def test_unknown_preset_raises():
    config = AppConfig.load(ROOT / "appsettings.json")
    with pytest.raises(KeyError):
        config.preset("NOPE")


def test_paths_are_relative_to_config_dir():
    config = AppConfig.load(ROOT / "appsettings.json")
    assert config.path("./data_cache") == ROOT / "data_cache"
