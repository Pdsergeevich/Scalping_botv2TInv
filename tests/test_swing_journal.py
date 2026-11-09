"""Тесты журнала сделок свинг-корзины (JSONL: где вход, где выход)."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from bot_a.engine.breakout_fsm import BreakoutEvent, BreakoutExitReason, BreakoutParams, BreakoutSide
from bot_a.models.domain import InstrumentMetadata, MarketKind
from bot_a.services.swing_runner import SwingBasketRunner, SwingBasketSettings
from bot_a.utils.time_provider import MSK


def _meta() -> InstrumentMetadata:
    return InstrumentMetadata(
        instrument_uid="TEST",
        ticker="TEST_CONT",
        class_code="SPBFUT",
        market=MarketKind.FUTURES,
        min_price_increment=Decimal("1"),
        min_price_increment_amount=Decimal("1"),
        lot_size=1,
        initial_margin_buy=Decimal("1000"),
        initial_margin_sell=Decimal("1000"),
        name="TEST",
    )


def _runner(path: Path) -> SwingBasketRunner:
    params = BreakoutParams().overrides(take_profit_atr=0.0, min_efficiency_ratio=0.0)
    return SwingBasketRunner(
        [(_meta(), params)],
        SwingBasketSettings(capital_rub=50_000.0, journal_path=path, dry_run=True),
    )


def _now_ms() -> int:
    return int(datetime.now(tz=MSK).timestamp() * 1000)


def test_journal_records_entry_and_exit(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    runner = _runner(path)
    assert path.exists(), "файл журнала создаётся сразу при старте (можно tail -f)"
    start = _now_ms()

    machine = runner._machines["TEST_CONT"]
    from bot_a.engine.breakout_fsm import BreakoutPosition

    machine.position = BreakoutPosition(
        side=BreakoutSide.LONG,
        quantity=1,
        entry_price=Decimal("100"),
        entry_ms=start,
        atr_points=1.0,
        stop_price=Decimal("98"),
        initial_stop=Decimal("98"),
        fee_round_trip_rub=Decimal("1"),
        max_price=Decimal("100"),
    )
    runner._open(
        "TEST_CONT",
        BreakoutEvent(kind="entry", time_ms=start, side=BreakoutSide.LONG, price=Decimal("100")),
    )
    runner._close(
        "TEST_CONT",
        BreakoutEvent(
            kind="exit",
            time_ms=start + 3_600_000,
            side=BreakoutSide.LONG,
            price=Decimal("104"),
            reason=BreakoutExitReason.TAKE_PROFIT,
        ),
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert [row["kind"] for row in rows] == ["entry", "exit"]
    entry, exit_ = rows
    assert entry["ticker"] == "TEST_CONT"
    assert entry["side"] == "Long"
    assert entry["qty"] >= 1
    assert entry["stop"] and entry["price"]
    assert exit_["reason"] == "TakeProfit"
    assert exit_["entryMs"] == start
    assert exit_["holdMin"] > 0
    assert exit_["netRub"] is not None
    assert entry["dryRun"] is True


def test_journal_disabled_without_path(tmp_path: Path) -> None:
    runner = _runner(None)  # type: ignore[arg-type]
    assert runner._journal is None
    runner._write_journal("entry", "TEST_CONT", time_ms=_now_ms())  # не должно падать


def test_journal_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "journal.jsonl"
    runner = _runner(path)
    runner._write_journal("exit", "TEST_CONT", time_ms=_now_ms(), netRub=1.0)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["kind"] == "exit"


def test_appsettings_uses_60m_bars() -> None:
    """v4.5 торгует на 60-минутных барах; CLI без флага обязан взять это из конфига."""
    from bot_a.config import AppConfig

    root = Path(__file__).resolve().parents[1]
    config = AppConfig.load(root / "appsettings.json")
    assert int(config.swing_setting("bar_minutes", 30)) == 60
