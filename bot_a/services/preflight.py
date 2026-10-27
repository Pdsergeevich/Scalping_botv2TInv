"""Предполётная проверка: доступы, счёт, ГО, метаданные инструментов."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Sequence, Tuple

from bot_a.broker import tinvest as broker
from bot_a.config import AppConfig, PairPreset
from bot_a.models.domain import ExecutionTopology, InstrumentMetadata
from bot_a.risk.guard import DailyKillSwitch, MoexSessionSchedule, RiskAndScheduleGuard
from bot_a.utils.quotation import fmt_rub
from bot_a.utils.time_provider import MSK

ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str
    warning: bool = False


@dataclass(frozen=True, slots=True)
class PreflightReport:
    preset_name: str
    account_id: str
    checks: Tuple[PreflightCheck, ...]
    leg1: Optional[InstrumentMetadata]
    leg2: Optional[InstrumentMetadata]
    free_cash_rub: Decimal
    blocked_cash_rub: Decimal
    required_margin_rub: Decimal
    tariff: str
    topology: str

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks if not check.warning)

    def render(self) -> str:
        lines = [
            f"Preflight: {self.preset_name}",
            "=" * 60,
            f"Счёт:            {self.account_id or '—'}",
            f"Тариф:           {self.tariff}",
            f"Топология:       {self.topology}",
            f"Свободные RUB:   {fmt_rub(self.free_cash_rub)}",
            f"Заблокировано:   {fmt_rub(self.blocked_cash_rub)}",
            f"Требуется ГО:    {fmt_rub(self.required_margin_rub)} (с буфером)",
            "",
            f"{'Проверка':<34}{'Статус':<10}Детали",
            "-" * 96,
        ]
        for check in self.checks:
            status = "OK" if check.ok else ("WARN" if check.warning else "FAIL")
            lines.append(f"{check.name:<34}{status:<10}{check.detail}")
        if self.leg1 is not None:
            lines.append("")
            lines.append(
                f"Leg1: {self.leg1.ticker} ({self.leg1.class_code}), uid={self.leg1.instrument_uid}, "
                f"шаг={self.leg1.min_price_increment}, стоимость шага={self.leg1.min_price_increment_amount}, "
                f"лот={self.leg1.lot_size}, ГО={fmt_rub(self.leg1.position_margin_rub(1))}"
            )
        if self.leg2 is not None:
            lines.append(
                f"Leg2: {self.leg2.ticker} ({self.leg2.class_code}), uid={self.leg2.instrument_uid}, "
                f"шаг={self.leg2.min_price_increment}, стоимость шага={self.leg2.min_price_increment_amount}, "
                f"лот={self.leg2.lot_size}, ГО={fmt_rub(self.leg2.position_margin_rub(1))}"
            )
        lines.append("")
        lines.append("ИТОГ: " + ("ГОТОВ К РАБОТЕ" if self.ok else "ЕСТЬ БЛОКИРУЮЩИЕ ОШИБКИ"))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "preset": self.preset_name,
            "accountId": self.account_id,
            "ok": self.ok,
            "tariff": self.tariff,
            "topology": self.topology,
            "freeCashRub": str(self.free_cash_rub),
            "blockedCashRub": str(self.blocked_cash_rub),
            "requiredMarginRub": str(self.required_margin_rub),
            "checks": [
                {"name": check.name, "ok": check.ok, "detail": check.detail, "warning": check.warning}
                for check in self.checks
            ],
            "leg1": self.leg1.to_dict() if self.leg1 else None,
            "leg2": self.leg2.to_dict() if self.leg2 else None,
        }


def _pick_account(accounts: Sequence[broker.AccountInfo], preferred: str) -> Optional[broker.AccountInfo]:
    if preferred:
        for account in accounts:
            if account.account_id == preferred:
                return account
        return None
    for account in accounts:
        if "FULL" in account.access_level.upper():
            return account
    return accounts[0] if accounts else None


def _required_margin(
    config: AppConfig,
    legs: Sequence[InstrumentMetadata],
    topology: ExecutionTopology,
) -> Decimal:
    quantity = max(int(config.strategy.quantity_lots), 1)
    guard = RiskAndScheduleGuard(
        MoexSessionSchedule(futures=any(meta.is_futures for meta in legs)),
        DailyKillSwitch(config.risk.max_daily_drawdown_rub),
        config.risk.margin_safety_buffer_pct,
        config.risk.max_position_lots,
    )
    quantity = guard.clamp_quantity(quantity)
    margin_legs = list(legs) if topology is ExecutionTopology.DUAL_LEG_HEDGED else [legs[0]]
    base = guard.required_margin_rub(margin_legs, quantity)
    return base * guard.margin_multiplier


async def run_preflight(
    config: AppConfig,
    preset: PairPreset,
    logger: Optional[logging.Logger] = None,
) -> PreflightReport:
    """Проверяет готовность счёта и резолвит метаданные обеих ног."""
    log = logger or logging.getLogger("bot_a.preflight")
    checks: List[PreflightCheck] = []
    token = config.t_invest.resolve_token(config.base_dir)
    checks.append(
        PreflightCheck(
            "Токен T-Invest",
            bool(token),
            f"задан ({config.t_invest.token_source(config.base_dir)})"
            if token
            else f"не задан (tInvest.token, ${config.t_invest.token_env_var} "
            f"или secrets/tinvest_token.txt)",
        )
    )
    if not token:
        return _report(config, preset, checks, preset.leg1, preset.leg2, ZERO, ZERO, ZERO, "")

    topology = config.execution.topology
    async with broker.async_client(
        token,
        config.t_invest.app_name,
        config.t_invest.request_timeout_seconds,
    ) as client:
        accounts = await broker.get_accounts(client)
        checks.append(
            PreflightCheck(
                "Доступ к счетам",
                bool(accounts),
                f"найдено счетов: {len(accounts)}" if accounts else "счета не найдены (проверьте токен и права)",
            )
        )
        account = _pick_account(accounts, config.t_invest.account_id)
        account_id = account.account_id if account else ""
        if account is None:
            checks.append(
                PreflightCheck(
                    "Выбор счёта",
                    False,
                    f"счёт '{config.t_invest.account_id}' не найден" if config.t_invest.account_id else "нет доступных счетов",
                )
            )
        else:
            full_access = "FULL" in account.access_level.upper()
            checks.append(
                PreflightCheck(
                    "Полный доступ к счёту",
                    full_access,
                    f"{account.name or account.account_id}: {account.access_level or 'unknown'}",
                    warning=not full_access,
                )
            )

        leg1, leg2, resolution_checks = await _resolve_legs(client, preset)
        checks.extend(resolution_checks)

        forts_legs = [meta for meta in (leg1, leg2) if meta.is_futures]
        if forts_legs:
            for meta in forts_legs:
                checks.append(
                    PreflightCheck(
                        f"ГО по {meta.ticker}",
                        meta.position_margin_rub(1) > 0,
                        f"InitialMargin={fmt_rub(meta.position_margin_rub(1))} RUB",
                    )
                )
                checks.append(
                    PreflightCheck(
                        f"Экономика шага {meta.ticker}",
                        meta.min_price_increment > 0 and meta.min_price_increment_amount > 0,
                        f"шаг={meta.min_price_increment} пунктов = {meta.min_price_increment_amount} RUB "
                        f"(1 пункт = {meta.rub_per_price_unit} RUB)",
                    )
                )

        for meta in (leg1, leg2):
            if not meta.instrument_uid:
                continue
            available, detail = await broker.is_trading_available(client, meta.instrument_uid)
            checks.append(
                PreflightCheck(
                    f"Торговый статус {meta.ticker}",
                    True,
                    ("торговля доступна; " if available else "торговля сейчас недоступна; ") + detail,
                    warning=not available,
                )
            )

        free_cash = ZERO
        blocked_cash = ZERO
        if account_id:
            try:
                free_cash, blocked_cash = await broker.get_positions_money(client, account_id)
                checks.append(
                    PreflightCheck(
                        "Свободные средства",
                        True,
                        f"{fmt_rub(free_cash)} RUB свободно, {fmt_rub(blocked_cash)} RUB заблокировано",
                    )
                )
            except Exception as exc:
                checks.append(PreflightCheck("Свободные средства", False, f"ошибка запроса: {exc}"))

        required_margin = _required_margin(config, [leg1, leg2], topology)
        if required_margin > 0:
            checks.append(
                PreflightCheck(
                    "Достаточность ГО",
                    free_cash >= required_margin,
                    f"нужно {fmt_rub(required_margin)} RUB (ГО + {config.risk.margin_safety_buffer_pct:.0f}% буфер), "
                    f"есть {fmt_rub(free_cash)} RUB",
                )
            )
        market_ok = all(meta.is_futures for meta in (leg1, leg2)) or all(
            not meta.is_futures for meta in (leg1, leg2)
        )
        checks.append(
            PreflightCheck(
                "Совместимость рынков пары",
                market_ok,
                "обе ноги на одном рынке" if market_ok else "ноги принадлежат разным рынкам (SPBFUT/TQBR)",
                warning=not market_ok,
            )
        )
        schedule = MoexSessionSchedule(futures=any(meta.is_futures for meta in (leg1, leg2)))
        blocking = schedule.blocking_reason(datetime.now(tz=MSK))
        checks.append(
            PreflightCheck(
                "Торговая сессия",
                blocking is None,
                blocking or "сессия открыта",
                warning=blocking is not None,
            )
        )
        report = _report(
            config,
            preset,
            checks,
            leg1,
            leg2,
            free_cash,
            blocked_cash,
            required_margin,
            account_id,
        )
        log.info("\n%s", report.render())
        return report


async def _resolve_legs_rest(
    client,
    preset: PairPreset,
) -> Tuple[InstrumentMetadata, InstrumentMetadata, List[PreflightCheck]]:
    """Резолв метаданных ног через REST (когда gRPC SDK недоступен)."""
    checks: List[PreflightCheck] = []
    leg1 = preset.leg1
    leg2 = preset.leg2
    for index, meta in ((1, preset.leg1), (2, preset.leg2)):
        try:
            resolved = await client.resolve_instrument(meta.ticker, meta.class_code, meta.market)
            checks.append(
                PreflightCheck(
                    f"Метаданные Leg{index} ({meta.ticker})",
                    bool(resolved.instrument_uid),
                    f"uid={resolved.instrument_uid}, lot={resolved.lot_size} (REST)",
                )
            )
            if index == 1:
                leg1 = resolved
            else:
                leg2 = resolved
        except Exception as exc:
            checks.append(
                PreflightCheck(
                    f"Метаданные Leg{index} ({meta.ticker})",
                    False,
                    f"не удалось получить через REST: {exc}",
                )
            )
    return leg1, leg2, checks


async def run_preflight_rest(
    config: AppConfig,
    preset: PairPreset,
    client,
    logger: Optional[logging.Logger] = None,
) -> PreflightReport:
    """Preflight через REST/JSON T-Invest (Python 3.14 без gRPC SDK).

    Проверяет токен, счета, метаданные инструментов, свободные средства,
    достаточность ГО и торговую сессию теми же правилами, что и gRPC-путь.
    """
    from bot_a.broker.tinkoff_rest_client import quotation_from_dict

    log = logger or logging.getLogger("bot_a.preflight")
    checks: List[PreflightCheck] = []
    token = config.t_invest.resolve_token(config.base_dir)
    checks.append(
        PreflightCheck(
            "Токен T-Invest",
            bool(token),
            f"задан ({config.t_invest.token_source(config.base_dir)}), транспорт REST",
        )
    )
    topology = config.execution.topology

    accounts_raw = await client.get_accounts()
    checks.append(
        PreflightCheck(
            "Доступ к счетам",
            bool(accounts_raw),
            f"найдено счетов: {len(accounts_raw)}"
            if accounts_raw
            else "счета не найдены (проверьте токен и права)",
        )
    )
    account = None
    preferred = config.t_invest.account_id
    for item in accounts_raw:
        acc_id = str(item.get("id") or item.get("accountId") or "")
        if preferred and acc_id == preferred:
            account = item
            break
    if account is None and accounts_raw:
        for item in accounts_raw:
            if "FULL" in str(item.get("accessLevel", "")).upper():
                account = item
                break
        if account is None:
            account = accounts_raw[0]
    account_id = str((account or {}).get("id") or (account or {}).get("accountId") or "")
    if account is None:
        checks.append(
            PreflightCheck(
                "Выбор счёта",
                False,
                f"счёт '{preferred}' не найден" if preferred else "нет доступных счетов",
            )
        )
    else:
        access = str(account.get("accessLevel", "") or "unknown")
        checks.append(
            PreflightCheck(
                "Полный доступ к счёту",
                "FULL" in access.upper(),
                f"{account.get('name') or account_id}: {access}",
                warning="FULL" not in access.upper(),
            )
        )

    leg1, leg2, resolution_checks = await _resolve_legs_rest(client, preset)
    checks.extend(resolution_checks)

    for meta in (leg1, leg2):
        if meta.is_futures:
            checks.append(
                PreflightCheck(
                    f"ГО по {meta.ticker}",
                    meta.position_margin_rub(1) > 0,
                    f"InitialMargin={fmt_rub(meta.position_margin_rub(1))} RUB",
                )
            )

    for index, meta in ((1, leg1), (2, leg2)):
        if not meta.instrument_uid:
            continue
        try:
            status = await client.get_trading_status(meta.instrument_uid)
            available = bool(status.get("marketOrderAvailable", True))
            detail = str(status.get("tradingStatus", "") or "статус не сообщён")
        except Exception as exc:  # статус не критичен для preflight
            available, detail = True, f"статус недоступен: {exc}"
        checks.append(
            PreflightCheck(
                f"Торговый статус {meta.ticker}", True,
                ("торговля доступна; " if available else "торговля сейчас недоступна; ") + detail,
                warning=not available,
            )
        )

    free_cash = ZERO
    blocked_cash = ZERO
    if account_id:
        try:
            positions = await client.get_positions(account_id)
            free_cash = await client.free_cash_rub(account_id)
            blocking = [
                quotation_from_dict(item)
                for item in (positions.get("blocked") or [])
                if str(item.get("currency", "")).lower() in ("rub", "rur")
            ]
            blocked_cash = sum(blocking, ZERO)
            checks.append(
                PreflightCheck(
                    "Свободные средства",
                    True,
                    f"{fmt_rub(free_cash)} RUB свободно, {fmt_rub(blocked_cash)} RUB заблокировано",
                )
            )
        except Exception as exc:
            checks.append(PreflightCheck("Свободные средства", False, f"ошибка запроса: {exc}"))

    required_margin = _required_margin(config, [leg1, leg2], topology)
    if required_margin > 0:
        checks.append(
            PreflightCheck(
                "Достаточность ГО",
                free_cash >= required_margin,
                f"нужно {fmt_rub(required_margin)} RUB (ГО + "
                f"{config.risk.margin_safety_buffer_pct:.0f}% буфер), есть {fmt_rub(free_cash)} RUB",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "Достаточность ГО",
                True,
                f"спот-пара: маржа не требуется, свободно {fmt_rub(free_cash)} RUB",
                warning=False,
            )
        )

    market_ok = all(meta.is_futures for meta in (leg1, leg2)) or all(
        not meta.is_futures for meta in (leg1, leg2)
    )
    checks.append(
        PreflightCheck(
            "Совместимость рынков пары",
            market_ok,
            "обе ноги на одном рынке" if market_ok else "ноги на разных рынках (SPBFUT/TQBR)",
            warning=not market_ok,
        )
    )
    schedule = MoexSessionSchedule(futures=any(meta.is_futures for meta in (leg1, leg2)))
    session_block = schedule.blocking_reason(datetime.now(tz=MSK))
    checks.append(
        PreflightCheck(
            "Торговая сессия",
            session_block is None,
            session_block or "сессия открыта",
            warning=session_block is not None,
        )
    )
    checks.append(
        PreflightCheck(
            "Каденция статистики",
            float(config.execution.stat_sample_interval_seconds) > 0,
            f"сэмпл раз в {config.execution.stat_sample_interval_seconds:.1f} с, "
            f"прогрев {config.execution.bootstrap_warmup_days} торг. дней",
            warning=float(config.execution.stat_sample_interval_seconds) <= 0,
        )
    )

    report = _report(
        config, preset, checks, leg1, leg2, free_cash, blocked_cash, required_margin, account_id
    )
    log.info("\n%s", report.render())
    return report


async def _resolve_legs(
    client,
    preset: PairPreset,
) -> Tuple[InstrumentMetadata, InstrumentMetadata, List[PreflightCheck]]:
    checks: List[PreflightCheck] = []
    leg1 = preset.leg1
    leg2 = preset.leg2
    for index, meta in ((1, preset.leg1), (2, preset.leg2)):
        try:
            resolved = await broker.resolve_instrument(client, meta.ticker, meta.class_code, meta.market)
            checks.append(
                PreflightCheck(
                    f"Метаданные Leg{index} ({meta.ticker})",
                    bool(resolved.instrument_uid),
                    f"uid={resolved.instrument_uid}, lot={resolved.lot_size}",
                )
            )
            if index == 1:
                leg1 = resolved
            else:
                leg2 = resolved
        except Exception as exc:
            checks.append(
                PreflightCheck(
                    f"Метаданные Leg{index} ({meta.ticker})",
                    False,
                    f"не удалось получить из API: {exc}",
                )
            )
    return leg1, leg2, checks


def _report(
    config: AppConfig,
    preset: PairPreset,
    checks: List[PreflightCheck],
    leg1: Optional[InstrumentMetadata],
    leg2: Optional[InstrumentMetadata],
    free_cash: Decimal,
    blocked_cash: Decimal,
    required_margin: Decimal,
    account_id: str,
) -> PreflightReport:
    return PreflightReport(
        preset_name=preset.name,
        account_id=account_id,
        checks=tuple(checks),
        leg1=leg1,
        leg2=leg2,
        free_cash_rub=free_cash,
        blocked_cash_rub=blocked_cash,
        required_margin_rub=required_margin,
        tariff=config.fees.tariff.value,
        topology=config.execution.topology.value,
    )
