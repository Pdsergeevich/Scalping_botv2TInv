"""Проверка готовности к торговой сессии («monday-ready»).

Офлайн-проверки, которые не требуют токена T-Invest: расписание MOEX и время до
ближайшего открытия, согласованность параметров v2.2, адекватность тарифа,
достаточность средств под ГО с буфером 25 %, разрешимость всех пресетов и
наличие данных/токена. Используется в ``--mode preflight --monday-ready``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Sequence, Tuple

from bot_a.config import AppConfig
from bot_a.risk.guard import MoexSessionSchedule
from bot_a.utils.quotation import fmt_rub
from bot_a.utils.time_provider import MSK

ZERO = Decimal("0")
_SCAN_LIMIT_MINUTES = 8 * 24 * 60


@dataclass(frozen=True, slots=True)
class MondayCheck:
    """Одна проверка готовности."""

    name: str
    ok: bool
    detail: str
    warning: bool = False


@dataclass(frozen=True, slots=True)
class MondayReadiness:
    """Сводка готовности к ближайшей сессии."""

    now_msk: datetime
    next_open_msk: Optional[datetime]
    minutes_to_open: Optional[int]
    checks: Tuple[MondayCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks if not check.warning)

    def render(self) -> str:
        lines = [
            "",
            "Готовность к сессии (monday-ready)",
            "=" * 60,
            f"Время MSK:       {self.now_msk:%Y-%m-%d %H:%M:%S} ({_WEEKDAYS[self.now_msk.weekday()]})",
        ]
        if self.next_open_msk is not None:
            lines.append(
                f"Открытие:        {self.next_open_msk:%Y-%m-%d %H:%M} MSK "
                f"(через {_humanize(self.minutes_to_open)})"
            )
        else:
            lines.append("Открытие:        не найдено в ближайшие 8 дней")
        lines.extend(["", f"{'Проверка':<34}{'Статус':<10}Детали", "-" * 96])
        for check in self.checks:
            status = "OK" if check.ok else ("WARN" if check.warning else "FAIL")
            lines.append(f"{check.name:<34}{status:<10}{check.detail}")
        lines.append("")
        lines.append("ИТОГ: " + ("К СЕССИИ ГОТОВ" if self.ok else "ЕСТЬ БЛОКИРУЮЩИЕ ПРОБЛЕМЫ"))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "nowMsk": self.now_msk.isoformat(),
            "nextOpenMsk": self.next_open_msk.isoformat() if self.next_open_msk else None,
            "minutesToOpen": self.minutes_to_open,
            "ok": self.ok,
            "checks": [
                {"name": item.name, "ok": item.ok, "detail": item.detail, "warning": item.warning}
                for item in self.checks
            ],
        }


_WEEKDAYS = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")


def _humanize(minutes: Optional[int]) -> str:
    if minutes is None:
        return "—"
    days, rest = divmod(int(minutes), 24 * 60)
    hours, mins = divmod(rest, 60)
    if days:
        return f"{days} д {hours} ч {mins} мин"
    if hours:
        return f"{hours} ч {mins} мин"
    return f"{mins} мин"


def next_session_open(schedule: MoexSessionSchedule, moment: datetime) -> Optional[datetime]:
    """Возвращает ближайший момент открытия сессии (с шагом в минуту)."""
    probe = moment.replace(second=0, microsecond=0)
    for _ in range(_SCAN_LIMIT_MINUTES):
        probe += timedelta(minutes=1)
        if schedule.is_open(probe):
            return probe
    return None


def _params_checks(config: AppConfig) -> list[MondayCheck]:
    params = config.strategy
    checks: list[MondayCheck] = []
    gap = float(params.z_stop) - float(params.z_entry)
    checks.append(
        MondayCheck(
            "Зазор Z_stop - Z_entry",
            gap >= 0.25,
            f"Z_stop={params.z_stop:.2f}, Z_entry={params.z_entry:.2f}, зазор={gap:.2f}",
        )
    )
    checks.append(
        MondayCheck(
            "Горизонт удержания",
            params.max_hold_seconds >= 900,
            f"MaxHold={params.max_hold_seconds} c, EWMA={params.ewma_window} тиков",
            warning=params.max_hold_seconds < 900,
        )
    )
    filters = {
        "хук пика Z": params.use_z_peak_hook,
        "режимный фильтр": params.use_regime_filter,
        "breakeven-ratchet": params.use_breakeven_ratchet,
        "сжатие тейка": params.use_tp_decay,
    }
    off = [name for name, enabled in filters.items() if not enabled]
    checks.append(
        MondayCheck(
            "Фильтры v2.2",
            not off,
            "все включены" if not off else f"выключены: {', '.join(off)}",
            warning=bool(off),
        )
    )
    ratchet_ok = (
        params.ratchet_exit_multiplier < params.ratchet_arm_multiplier
        and params.ratchet_arm_multiplier > params.min_net_profit_multiplier
    )
    checks.append(
        MondayCheck(
            "Калибровка ratchet",
            ratchet_ok,
            f"arm=max({params.ratchet_arm_multiplier}·Fee, {params.ratchet_arm_target_fraction:.2f}·цель)"
            f"+{params.ratchet_arm_step_buffer} шаг, exit={params.ratchet_exit_multiplier}·Fee "
            f"после отката {params.ratchet_retrace_steps} шагов и {params.ratchet_min_hold_seconds} c",
            warning=not ratchet_ok,
        )
    )
    return checks


def _tariff_check(config: AppConfig) -> MondayCheck:
    premium = config.fees.tariff.value.lower().startswith("premium")
    return MondayCheck(
        "Тариф брокера",
        premium,
        "Premium: 0.025 % за сторону — покрывает амплитуду спреда акций"
        if premium
        else "Trader: 0.04 % за сторону; на SBERP/T комиссия круга превышает типичный gross",
        warning=not premium,
    )


def _session_windows_check(config: AppConfig) -> MondayCheck:
    """v2.3: режим внутридневных окон входа."""
    from bot_a.risk.guard import IntradayEntryWindows

    mode = config.risk.active_session_windows
    windows = IntradayEntryWindows(mode)
    detail = {
        "AllDay": "торговля всю сессию (включая обед и вечер)",
        "MainSessionOnly": "входы 10:03–18:35 MSK",
        "PrimeHoursOnly": "входы 10:05–12:30 и 15:30–18:35 MSK",
    }.get(windows.mode, mode)
    return MondayCheck(
        "Окна входа (v2.3)",
        windows.mode != "AllDay",
        f"{windows.mode}: {detail}",
        warning=windows.mode == "AllDay",
    )


def _regime_checks(config: AppConfig) -> MondayCheck:
    """v2.4: режимные предохранители (дневной стоп, дрейф беты, σ/Fee)."""
    params = config.strategy
    enabled = {
        "дневной стоп": params.use_daily_stop_breaker,
        "дрейф беты": params.use_macro_beta_drift_guard,
        "σ/Fee": params.use_vol_to_fee_filter,
    }
    off = [name for name, on in enabled.items() if not on]
    detail = (
        f"HardStop ≤ {params.max_daily_hard_stops}/день или убыток ≤ "
        f"−{params.max_daily_loss_fee_multiple}·Fee; "
        f"дрейф беты ±{params.max_beta_drift * 100:.1f}% "
        f"(макро-окно {params.macro_beta_window} тиков); "
        f"σ_leg1/Fee ≥ {params.min_sigma_to_fee_ratio:.2f}"
    )
    if off:
        detail += f"; выключены: {', '.join(off)}"
    return MondayCheck(
        "Режимы v2.4",
        not off,
        detail,
        warning=bool(off),
    )


def _cadence_check(config: AppConfig) -> MondayCheck:
    """v2.4.1: каденция статистики и объём прогрева перед live/paper."""
    interval = float(config.execution.stat_sample_interval_seconds)
    days = int(config.execution.bootstrap_warmup_days)
    stats_seconds = config.strategy.ewma_window * interval if interval > 0 else 0.0
    macro_seconds = config.strategy.macro_beta_window * interval if interval > 0 else 0.0
    regime_seconds = config.strategy.regime_window * interval if interval > 0 else 0.0
    ok = interval > 0 and days > 0
    detail = (
        f"сэмпл раз в {interval:.1f} с: EWMA {stats_seconds / 60:.0f} мин, "
        f"макро-бета {macro_seconds / 3600:.1f} ч, режим {regime_seconds:.0f} с; "
        f"прогрев {days} торг. дней из MOEX ISS; стейт дня {config.storage.state_dir}"
    )
    if not ok:
        detail = (
            f"интервал {interval:.1f} с, прогрев {days} дн.: live/paper получат окна в тиках L2"
        )
    return MondayCheck("Каденция и прогрев (v2.4.1)", ok, detail, warning=not ok)


def _margin_checks(config: AppConfig) -> list[MondayCheck]:
    checks: list[MondayCheck] = []
    buffer = Decimal(1) + Decimal(str(config.risk.margin_safety_buffer_pct)) / Decimal(100)
    fallback_quantity = max(int(config.strategy.quantity_lots), 1)
    cash = config.execution.paper_initial_cash_rub
    worst_name = "—"
    worst_margin = ZERO
    for name in config.trading_preset_names():
        preset = config.preset(name)
        # v2.6.2: у каждой пары свой размер позиции — ГО считается по её лотам,
        # а не по глобальному значению (иначе 55 лотов VTBR «раздували» ГО фьючерса).
        quantity = max(int(preset.quantity_lots or fallback_quantity), 1)
        legs = [preset.leg1, preset.leg2] if config.execution.topology.value.startswith("Dual") else [preset.leg1]
        required = sum((meta.position_margin_rub(quantity) for meta in legs), ZERO) * buffer
        if required > worst_margin:
            worst_name, worst_margin = name, required
        checks.append(
            MondayCheck(
                f"ГО: {name}",
                required <= cash,
                f"{fmt_rub(required)} RUB с буфером {config.risk.margin_safety_buffer_pct:.0f} % "
                f"({quantity} лот., свободно {fmt_rub(cash)})",
            )
        )
    if worst_margin <= cash:
        checks.append(
            MondayCheck("Максимальное ГО", True, f"{worst_name}: {fmt_rub(worst_margin)} RUB")
        )
    return checks


def _preset_check(config: AppConfig) -> MondayCheck:
    names = config.trading_preset_names()
    broken = []
    for name in names:
        preset = config.preset(name)
        if preset.beta_mode not in ("static", "same_underlying", "slow_ratio", "dynamic"):
            broken.append(name)
    disabled = config.disabled_preset_names()
    detail = f"активных {len(names)}: {', '.join(names)}" if not broken else f"некорректный betaMode: {broken}"
    if disabled:
        detail += f"; отключены: {', '.join(disabled)}"
    return MondayCheck(
        "Пресеты и betaMode",
        bool(names) and not broken,
        detail,
    )


def _data_check(config: AppConfig, logger: logging.Logger) -> MondayCheck:
    token = config.t_invest.resolve_token(config.base_dir)
    cache_dir = config.path(config.storage.data_cache_dir)
    cached = len(list(cache_dir.glob("iss_*.json"))) if cache_dir.exists() else 0
    if token:
        return MondayCheck(
            "Данные и доступ",
            True,
            f"токен T-Invest задан ({config.t_invest.token_source(config.base_dir)}), "
            f"кэш ISS: {cached} файлов",
        )
    return MondayCheck(
        "Данные и доступ",
        True,
        f"токена нет: live недоступен, paper пойдёт через MOEX ISS (--source iss) "
        f"или REST после добавления secrets/tinvest_token.txt; кэш ISS: {cached} файлов",
        warning=True,
    )


def build_monday_readiness(
    config: AppConfig,
    now: Optional[datetime] = None,
    logger: Optional[logging.Logger] = None,
) -> MondayReadiness:
    """Собирает все проверки готовности к ближайшей торговой сессии."""
    log = logger or logging.getLogger("bot_a.readiness")
    moment = (now or datetime.now(MSK)).astimezone(MSK)
    config = config.with_preset_overrides()
    futures = config.preset().leg1.is_futures
    schedule = MoexSessionSchedule(
        futures=futures, trade_evening_session=config.risk.trade_evening_session
    )
    blocking = schedule.blocking_reason(moment)
    next_open = None if blocking is None else next_session_open(schedule, moment)
    minutes = None
    if next_open is not None:
        minutes = int((next_open - moment).total_seconds() // 60)
    checks: list[MondayCheck] = [
        MondayCheck(
            "Расписание MOEX",
            blocking is None,
            "сессия открыта" if blocking is None else blocking,
            warning=blocking is not None,
        )
    ]
    checks.extend(_params_checks(config))
    checks.append(_tariff_check(config))
    checks.append(_session_windows_check(config))
    checks.append(_regime_checks(config))
    checks.append(_cadence_check(config))
    checks.extend(_margin_checks(config))
    checks.append(_preset_check(config))
    checks.append(_data_check(config, log))
    return MondayReadiness(
        now_msk=moment,
        next_open_msk=next_open,
        minutes_to_open=minutes,
        checks=tuple(checks),
    )
