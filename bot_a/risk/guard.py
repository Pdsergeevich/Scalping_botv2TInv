"""Риск-контроль: расписание MOEX, клиринг, ГО и дневной kill-switch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Optional, Sequence, Tuple

from bot_a.models.domain import InstrumentMetadata, MarketKind
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import ms_to_msk, trading_day_key

ZERO = Decimal("0")


def _minutes(hour: int, minute: int) -> int:
    return hour * 60 + minute


@dataclass(frozen=True, slots=True)
class SessionWindow:
    """Торговое окно в минутах от полуночи MSK."""

    start_minute: int
    end_minute: int

    def contains(self, minute_of_day: float) -> bool:
        return self.start_minute <= minute_of_day < self.end_minute


@dataclass(frozen=True, slots=True)
class GuardDecision:
    allowed: bool
    reason: str = ""

    @staticmethod
    def ok() -> "GuardDecision":
        return GuardDecision(True, "ok")

    @staticmethod
    def deny(reason: str) -> "GuardDecision":
        return GuardDecision(False, reason)


class MoexSessionSchedule:
    """Основная и вечерняя сессии MOEX с технологическими перерывами."""
    CLEARING_WINDOWS: Tuple[SessionWindow, ...] = (
        SessionWindow(_minutes(14, 0), _minutes(14, 5)),
        SessionWindow(_minutes(18, 45), _minutes(19, 5)),
    )

    def __init__(self, futures: bool, trade_evening_session: bool = True) -> None:
        self._futures = futures
        self._trade_evening = trade_evening_session
        if futures:
            # Срочный рынок: 10:00–14:00, 14:05–18:45, 19:05–23:50.
            self._windows: Tuple[SessionWindow, ...] = (
                SessionWindow(_minutes(10, 0), _minutes(14, 0)),
                SessionWindow(_minutes(14, 5), _minutes(18, 45)),
                SessionWindow(_minutes(19, 5), _minutes(23, 50)),
            )
        else:
            # Фондовый рынок TQBR: 09:50–14:00, 14:05–18:40, 19:05–23:50.
            self._windows = (
                SessionWindow(_minutes(9, 50), _minutes(14, 0)),
                SessionWindow(_minutes(14, 5), _minutes(18, 40)),
                SessionWindow(_minutes(19, 5), _minutes(23, 50)),
            )

    @property
    def futures(self) -> bool:
        return self._futures

    def blocking_reason(self, moment: datetime) -> Optional[str]:
        if moment.weekday() >= 5:
            return "выходной день (MOEX не торгует)"
        minute_of_day = moment.hour * 60 + moment.minute + moment.second / 60.0
        for window in self.CLEARING_WINDOWS:
            if window.contains(minute_of_day):
                return (
                    "технологический перерыв/клиринг "
                    f"{window.start_minute // 60:02d}:{window.start_minute % 60:02d}"
                    f"–{window.end_minute // 60:02d}:{window.end_minute % 60:02d} MSK"
                )
        if not self._trade_evening and minute_of_day >= _minutes(18, 45):
            return "вечерняя сессия отключена в конфигурации"
        for window in self._windows:
            if window.contains(minute_of_day):
                return None
        return "вне торговой сессии MOEX"

    def is_open(self, moment: datetime) -> bool:
        return self.blocking_reason(moment) is None


class IntradayEntryWindows:
    """Активные внутридневные окна для новых входов (MSK).

    Выходы и аварийные закрытия ограничений не имеют — фильтр действует только
    на открытие новых позиций, чтобы не торговать в «мёртвые» часы.
    """

    ALL_DAY = "AllDay"
    MAIN_SESSION_ONLY = "MainSessionOnly"
    PRIME_HOURS_ONLY = "PrimeHoursOnly"
    MODES: Tuple[str, ...] = (ALL_DAY, MAIN_SESSION_ONLY, PRIME_HOURS_ONLY)

    def __init__(self, mode: str = ALL_DAY) -> None:
        self._mode = mode if mode in self.MODES else self.ALL_DAY
        if self._mode == self.MAIN_SESSION_ONLY:
            self._windows: Tuple[SessionWindow, ...] = (SessionWindow(_minutes(10, 3), _minutes(18, 35)),)
        elif self._mode == self.PRIME_HOURS_ONLY:
            self._windows = (
                SessionWindow(_minutes(10, 5), _minutes(12, 30)),
                SessionWindow(_minutes(15, 30), _minutes(18, 35)),
            )
        else:
            self._windows = ()

    @property
    def mode(self) -> str:
        return self._mode

    def allows(self, moment: datetime) -> bool:
        if not self._windows:
            return True
        minute_of_day = moment.hour * 60 + moment.minute + moment.second / 60.0
        return any(window.contains(minute_of_day) for window in self._windows)

    def blocking_reason(self, moment: datetime) -> Optional[str]:
        if self.allows(moment):
            return None
        if self._mode == self.MAIN_SESSION_ONLY:
            return "вне основной сессии для входа (10:03–18:35 MSK)"
        return "вне активных часов для входа (10:05–12:30 и 15:30–18:35 MSK)"


class DailyKillSwitch:
    """Останавливает торговлю при превышении дневного лимита убытка."""

    def __init__(self, max_daily_drawdown_rub: Decimal) -> None:
        self._limit = abs(dec(max_daily_drawdown_rub))
        self._day: Optional[date] = None
        self._pnl = ZERO

    @property
    def limit_rub(self) -> Decimal:
        return self._limit

    @property
    def daily_pnl_rub(self) -> Decimal:
        return self._pnl

    def _roll(self, day: date) -> None:
        if self._day != day:
            self._day = day
            self._pnl = ZERO

    def register_pnl(self, pnl_rub: Decimal, day: date) -> None:
        self._roll(day)
        self._pnl += dec(pnl_rub)

    def is_triggered(self, day: date) -> bool:
        self._roll(day)
        if self._limit <= 0:
            return False
        return self._pnl <= -self._limit

    def reset(self) -> None:
        self._day = None
        self._pnl = ZERO


class RiskAndScheduleGuard:
    """Единая точка проверки: расписание + kill-switch + маржинальный буфер."""

    def __init__(
        self,
        schedule: MoexSessionSchedule,
        kill_switch: DailyKillSwitch,
        margin_safety_buffer_pct: float = 25.0,
        max_position_lots: int = 0,
        entry_windows: Optional[IntradayEntryWindows] = None,
    ) -> None:
        self._schedule = schedule
        self._kill_switch = kill_switch
        self._buffer_pct = max(float(margin_safety_buffer_pct), 0.0)
        self._max_lots = max(int(max_position_lots), 0)
        self._entry_windows = entry_windows or IntradayEntryWindows()

    @property
    def entry_windows(self) -> IntradayEntryWindows:
        return self._entry_windows

    def entry_window_reason(self, now_ms: int) -> Optional[str]:
        """Причина запрета входа по внутридневному фильтру (None — вход разрешён)."""
        return self._entry_windows.blocking_reason(ms_to_msk(now_ms))

    @property
    def schedule(self) -> MoexSessionSchedule:
        return self._schedule

    @property
    def kill_switch(self) -> DailyKillSwitch:
        return self._kill_switch

    @property
    def margin_multiplier(self) -> Decimal:
        return Decimal(1) + Decimal(str(self._buffer_pct / 100.0))

    def clamp_quantity(self, requested: int) -> int:
        """Ограничивает размер позиции настройкой ``maxPositionLots``."""
        lots = max(int(requested), 1)
        if self._max_lots > 0:
            return min(lots, self._max_lots)
        return lots

    def required_margin_rub(self, legs: Sequence[InstrumentMetadata], quantity: int) -> Decimal:
        total = ZERO
        for meta in legs:
            total += meta.position_margin_rub(quantity)
        return total

    def evaluate(
        self,
        now_ms: int,
        free_cash_rub: Decimal,
        required_margin_rub: Decimal,
    ) -> GuardDecision:
        moment = ms_to_msk(now_ms)
        reason = self._schedule.blocking_reason(moment)
        if reason is not None:
            return GuardDecision.deny(reason)
        day = trading_day_key(now_ms)
        if self._kill_switch.is_triggered(day):
            return GuardDecision.deny(
                f"дневной лимит убытка исчерпан: {self._kill_switch.daily_pnl_rub} RUB"
                f" (лимит {self._kill_switch.limit_rub} RUB)"
            )
        if required_margin_rub > 0:
            need = dec(required_margin_rub) * self.margin_multiplier
            if dec(free_cash_rub) < need:
                return GuardDecision.deny(
                    f"недостаточно свободных средств: {free_cash_rub} RUB < {need} RUB "
                    f"(ГО {required_margin_rub} + буфер {self._buffer_pct:.0f}%)"
                )
        return GuardDecision.ok()

    def register_trade_pnl(self, pnl_rub: Decimal, closed_ms: int) -> None:
        self._kill_switch.register_pnl(pnl_rub, trading_day_key(closed_ms))

    def can_flatten(self, now_ms: int) -> GuardDecision:
        """Аварийное закрытие разрешено всегда, кроме клиринга (заявки не пройдут)."""
        moment = ms_to_msk(now_ms)
        for window in MoexSessionSchedule.CLEARING_WINDOWS:
            minute_of_day = moment.hour * 60 + moment.minute + moment.second / 60.0
            if window.contains(minute_of_day):
                return GuardDecision.deny("клиринг: аварийное закрытие невозможно")
        return GuardDecision.ok()

    def build_for_markets(self, legs: Sequence[InstrumentMetadata], trade_evening_session: bool) -> "RiskAndScheduleGuard":
        futures = any(meta.market is MarketKind.FUTURES for meta in legs)
        schedule = MoexSessionSchedule(futures=futures, trade_evening_session=trade_evening_session)
        return RiskAndScheduleGuard(schedule, self._kill_switch, float(self._buffer_pct), self._max_lots)
