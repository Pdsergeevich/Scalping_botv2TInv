"""Проверки расписания MOEX, kill-switch и маржинального буфера."""

from datetime import datetime, timedelta
from decimal import Decimal

from bot_a.risk.guard import DailyKillSwitch, MoexSessionSchedule, RiskAndScheduleGuard
from bot_a.utils.time_provider import MSK, msk_to_ms
from tests.helpers import make_meta


def _moment(day: str, hour: int, minute: int) -> datetime:
    date = datetime.fromisoformat(day).date()
    return datetime(date.year, date.month, date.day, hour, minute, tzinfo=MSK)


def test_futures_session_windows():
    schedule = MoexSessionSchedule(futures=True)
    assert schedule.is_open(_moment("2025-03-11", 10, 30))
    assert schedule.is_open(_moment("2025-03-11", 15, 0))
    assert schedule.is_open(_moment("2025-03-11", 20, 0))
    assert not schedule.is_open(_moment("2025-03-11", 9, 0))
    assert not schedule.is_open(_moment("2025-03-11", 14, 2))  # клиринг
    assert not schedule.is_open(_moment("2025-03-11", 18, 50))  # клиринг
    assert not schedule.is_open(_moment("2025-03-15", 12, 0))  # суббота
    assert "клиринг" in (schedule.blocking_reason(_moment("2025-03-11", 14, 1)) or "")


def test_evening_session_can_be_disabled():
    schedule = MoexSessionSchedule(futures=True, trade_evening_session=False)
    assert not schedule.is_open(_moment("2025-03-11", 20, 0))
    assert schedule.is_open(_moment("2025-03-11", 12, 0))


def test_daily_kill_switch():
    switch = DailyKillSwitch(Decimal("1000"))
    day = _moment("2025-03-11", 12, 0).date()
    switch.register_pnl(Decimal("-400"), day)
    assert not switch.is_triggered(day)
    switch.register_pnl(Decimal("-700"), day)
    assert switch.is_triggered(day)
    next_day = day + timedelta(days=1)
    assert not switch.is_triggered(next_day)
    assert switch.daily_pnl_rub == Decimal("0")


def test_margin_and_schedule_evaluate():
    meta = make_meta("AAA", margin="1000")
    guard = RiskAndScheduleGuard(
        MoexSessionSchedule(futures=True),
        DailyKillSwitch(Decimal("5000")),
        margin_safety_buffer_pct=25.0,
    )
    now = msk_to_ms(_moment("2025-03-11", 12, 0))
    required = guard.required_margin_rub([meta], 1)
    assert required == Decimal("1000")
    assert guard.evaluate(now, Decimal("1200"), required).allowed is False
    assert guard.evaluate(now, Decimal("1300"), required).allowed is True
    blocked = msk_to_ms(_moment("2025-03-11", 14, 2))
    decision = guard.evaluate(blocked, Decimal("100000"), required)
    assert decision.allowed is False
    assert "клиринг" in decision.reason


def test_kill_switch_blocks_entries():
    guard = RiskAndScheduleGuard(
        MoexSessionSchedule(futures=True),
        DailyKillSwitch(Decimal("500")),
        margin_safety_buffer_pct=0.0,
    )
    now = msk_to_ms(_moment("2025-03-11", 12, 0))
    guard.register_trade_pnl(Decimal("-600"), now)
    decision = guard.evaluate(now, Decimal("100000"), Decimal("0"))
    assert decision.allowed is False
    assert "лимит убытка" in decision.reason
