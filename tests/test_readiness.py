"""Тесты проверки готовности к сессии (monday-ready)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path

from bot_a.config import AppConfig
from bot_a.economics.fees import Tariff
from bot_a.risk.guard import MoexSessionSchedule
from bot_a.services.readiness import build_monday_readiness, next_session_open
from bot_a.utils.time_provider import MSK

ROOT = Path(__file__).resolve().parents[1]


def _config() -> AppConfig:
    return AppConfig.load(ROOT / "appsettings.json")


def test_next_session_open_skips_weekend():
    schedule = MoexSessionSchedule(futures=True)
    saturday = datetime(2026, 10, 24, 12, 0, tzinfo=MSK)
    assert next_session_open(schedule, saturday) == datetime(2026, 10, 26, 10, 0, tzinfo=MSK)


def test_next_session_open_inside_clearing_gap():
    schedule = MoexSessionSchedule(futures=False)
    clearing = datetime(2026, 10, 20, 14, 1, tzinfo=MSK)
    assert next_session_open(schedule, clearing) == datetime(2026, 10, 20, 14, 5, tzinfo=MSK)


def test_monday_readiness_reports_next_open_and_warnings():
    config = _config()
    ready = build_monday_readiness(config, now=datetime(2026, 10, 25, 13, 56, tzinfo=MSK))
    # Активный пресет VTBR_SBER — спот TQBR, открытие в 09:50 MSK.
    assert ready.next_open_msk == datetime(2026, 10, 26, 9, 50, tzinfo=MSK)
    assert ready.minutes_to_open is not None and ready.minutes_to_open > 0
    checks = {item.name: item for item in ready.checks}
    assert checks["Расписание MOEX"].warning is True  # воскресенье
    assert checks["Расписание MOEX"].ok is False
    assert checks["Зазор Z_stop - Z_entry"].ok is True
    # v2.3: ratchet по умолчанию выключен, поэтому чек «Фильтры v2.2» даёт предупреждение.
    assert checks["Фильтры v2.2"].warning is True
    assert "breakeven-ratchet" in checks["Фильтры v2.2"].detail
    assert checks["Калибровка ratchet"].ok is True
    assert "отката 2 шагов" in checks["Калибровка ratchet"].detail
    assert checks["Окна входа (v2.3)"].ok is True
    assert "PrimeHoursOnly" in checks["Окна входа (v2.3)"].detail
    assert checks["Тариф брокера"].ok is False and checks["Тариф брокера"].warning is True
    assert checks["Пресеты и betaMode"].ok is True
    assert "отключены: CNYRUBF_CRZ6" in checks["Пресеты и betaMode"].detail
    assert ready.ok is True  # предупреждения не блокируют


def test_monday_readiness_premium_is_ok_and_short_gap_fails():
    config = _config().with_overrides(tariff=Tariff.PREMIUM)
    assert config.strategy.z_stop - config.strategy.z_entry >= 0.25
    short_gap = config.with_overrides(strategy=config.strategy.with_overrides(z_stop=2.95))
    ready = build_monday_readiness(short_gap, now=datetime(2026, 10, 26, 12, 0, tzinfo=MSK))
    checks = {item.name: item for item in ready.checks}
    assert checks["Тариф брокера"].ok is True
    assert checks["Зазор Z_stop - Z_entry"].ok is False
    assert ready.ok is False


def test_monday_readiness_margin_covers_every_active_preset():
    config = _config()
    ready = build_monday_readiness(config, now=datetime(2026, 10, 26, 12, 0, tzinfo=MSK))
    margin_checks = [item for item in ready.checks if item.name.startswith("ГО: ")]
    assert len(margin_checks) == len(config.trading_preset_names())
    assert all(item.ok for item in margin_checks)
    assert not any("CNYRUBF_CRZ6" in item.name for item in margin_checks)
    spz = next(item for item in margin_checks if "SPZ6_SRZ6" in item.name)
    expected = config.preset("SPZ6_SRZ6").leg1.position_margin_rub(1) * Decimal("1.25")
    assert spz.ok
    assert "25" in spz.detail
    assert round(float(expected), 2) > 1000.0


def test_disabled_preset_excluded_from_trading_set():
    config = _config()
    assert "CNYRUBF_CRZ6" in config.preset_names()
    assert "CNYRUBF_CRZ6" not in config.trading_preset_names()
    assert config.preset("CNYRUBF_CRZ6").enabled is False
    assert config.preset("VTBR_SBER").enabled is True
