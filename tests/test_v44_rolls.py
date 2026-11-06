"""Тесты v4.4: back-adjustment роллов, барьеры переключения, анти-застой, лимит ГО.

Проверяем ровно те ошибки, из-за которых v4.3 терял деньги:
фантомные скачки на стыке контрактов, сделки «через ролл», откат прибыли
в застое и голодание лёгких пар из-за тяжёлого ГО одной позиции.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from bot_a.backtest import rolls as roll_map
from bot_a.backtest.candles import Candle, resample_candles
from bot_a.engine.breakout_fsm import (
    Bar,
    BreakoutParams,
    BreakoutPosition,
    BreakoutSide,
    IndicatorSnapshot,
    MomentumBreakoutStateMachine,
    BreakoutExitReason,
)
from bot_a.models.domain import InstrumentMetadata, MarketKind
from bot_a.utils.time_provider import MSK

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

DAY_MS = 86_400_000


def _ms(text: str, hour: int = 10) -> int:
    moment = datetime.fromisoformat(text).replace(hour=hour, tzinfo=MSK)
    return int(moment.timestamp() * 1000)


def _meta(ticker: str = "TEST_CONT", step: str = "1", step_price: str = "1", margin: int = 1000):
    return InstrumentMetadata(
        instrument_uid=ticker,
        ticker=ticker,
        class_code="SPBFUT",
        market=MarketKind.FUTURES,
        min_price_increment=Decimal(step),
        min_price_increment_amount=Decimal(step_price),
        lot_size=1,
        initial_margin_buy=Decimal(margin),
        initial_margin_sell=Decimal(margin),
        name=ticker,
    )


# ---------------------------------------------------------------- rolls


def test_load_rolls_missing_and_present(tmp_path: Path) -> None:
    assert roll_map.load_rolls("NOPE", tmp_path / "none.json") == []
    path = tmp_path / "rolls.json"
    path.write_text(
        json.dumps({"GZR_CONT": [{"day": "20260618", "from": "GZM6", "to": "GZU6", "delta": 397.0}]}),
        encoding="utf-8",
    )
    assert roll_map.roll_days("GZR_CONT", path) == ["20260618"]
    assert roll_map.load_rolls("OTHER", path) == []


def test_boundary_ms_is_last_bar_before_roll(tmp_path: Path) -> None:
    path = tmp_path / "rolls.json"
    path.write_text(
        json.dumps({"T_CONT": [{"day": "20260618", "from": "A", "to": "B", "delta": 10.0}]}),
        encoding="utf-8",
    )
    candles = [
        Candle(_ms("2026-06-16"), Decimal("100"), Decimal("100"), Decimal("100"), Decimal("100"), 1),
        Candle(_ms("2026-06-17", 18), Decimal("101"), Decimal("101"), Decimal("101"), Decimal("101"), 1),
        Candle(_ms("2026-06-18"), Decimal("110"), Decimal("110"), Decimal("110"), Decimal("110"), 1),
    ]
    boundaries = roll_map.boundary_ms("T_CONT", candles, path)
    assert boundaries == {_ms("2026-06-17", 18)}


def test_boundary_ms_empty_without_rolls(tmp_path: Path) -> None:
    candles = [Candle(_ms("2026-06-16"), Decimal("1"), Decimal("1"), Decimal("1"), Decimal("1"), 1)]
    assert roll_map.boundary_ms("T_CONT", candles, tmp_path / "none.json") == set()


# ---------------------------------------------------------------- back-adjustment


def test_stitch_back_adjusts_roll_gap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Спред ролла 5 пт не должен остаться в склеенном ряду скачком."""
    import discover_forts_universe as dfu

    monkeypatch.setattr(dfu, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(dfu, "ROLLS_CACHE", tmp_path / "forts_rolls.json")

    # Два контракта одного семейства: 16–17.06 торгуется OLD, 18.06 — NEW,
    # на дне ролла оба живы и разница ровно 5 пунктов.
    old_rows = [
        {"time": _ms("2026-06-16"), "open": "100", "high": "101", "low": "99", "close": "100", "volume": 10},
        {"time": _ms("2026-06-17"), "open": "100", "high": "101", "low": "99", "close": "100", "volume": 10},
        {"time": _ms("2026-06-18", 9), "open": "100", "high": "100", "low": "100", "close": "100", "volume": 1},
    ]
    new_rows = [
        {"time": _ms("2026-06-18", 9), "open": "105", "high": "105", "low": "105", "close": "105", "volume": 100},
        {"time": _ms("2026-06-18", 10), "open": "105", "high": "106", "low": "104", "close": "105", "volume": 100},
        {"time": _ms("2026-06-18", 11), "open": "105", "high": "106", "low": "104", "close": "105", "volume": 100},
    ]
    for day, rows in (("20260616", old_rows[:1]), ("20260617", old_rows[1:2]),
                      ("20260618", old_rows[2:])):
        (tmp_path / f"iss_forts_OLDM6_{day}.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "iss_forts_NEWU6_20260618.json").write_text(json.dumps(new_rows), encoding="utf-8")

    universe = {
        "series": {
            "OLDM6": {"asset": "TEST", "margin": 1000, "reason": "front"},
            "NEWU6": {"asset": "TEST", "margin": 1000, "reason": "front"},
        }
    }
    dfu.stitch(universe)

    path = tmp_path / "forts_rolls.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["TEST_CONT"][0]["from"] == "OLDM6"
    assert saved["TEST_CONT"][0]["to"] == "NEWU6"
    assert saved["TEST_CONT"][0]["delta"] == pytest.approx(5.0)

    series = []
    for day in ("20260616", "20260617", "20260618"):
        rows = json.loads((tmp_path / f"iss_forts_TEST_CONT_{day}.json").read_text(encoding="utf-8"))
        series.extend(rows)
    series.sort(key=lambda item: item["time"])
    # Старая часть сдвинута на +5: 100 -> 105 (без фантомного скачка).
    assert float(series[0]["close"]) == pytest.approx(105.0)
    assert float(series[0]["adj"]) == pytest.approx(5.0)
    assert float(series[-1]["adj"]) == pytest.approx(0.0)
    closes = [float(item["close"]) for item in series]
    assert max(abs(b - a) for a, b in zip(closes, closes[1:])) <= 1.0


def test_candle_keeps_adj_through_resample() -> None:
    first = Candle(_ms("2026-06-18", 10), Decimal("100"), Decimal("101"), Decimal("99"),
                   Decimal("100"), 5, adj=7.0)
    second = Candle(_ms("2026-06-18", 10) + 60_000, Decimal("100"), Decimal("102"),
                    Decimal("99"), Decimal("101"), 5, adj=7.0)
    resampled = resample_candles([first, second], 5)
    assert resampled[0].adj == pytest.approx(7.0)
    restored = Candle.from_dict(resampled[0].to_dict())
    assert restored.adj == pytest.approx(7.0)


# ---------------------------------------------------------------- анти-застой


def _machine(**overrides) -> MomentumBreakoutStateMachine:
    params = BreakoutParams().overrides(**overrides)
    return MomentumBreakoutStateMachine(_meta(), params)


def _bar(time_ms: int, price: str, high: str, low: str) -> Bar:
    return Bar(
        time_ms=time_ms,
        open=Decimal(price),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(price),
        volume=100,
    )


def _position(machine: MomentumBreakoutStateMachine) -> BreakoutPosition:
    position = BreakoutPosition(
        side=BreakoutSide.LONG,
        quantity=1,
        entry_price=Decimal("100"),
        entry_ms=_ms("2026-06-18"),
        atr_points=1.0,
        stop_price=Decimal("98"),
        initial_stop=Decimal("98"),
        fee_round_trip_rub=Decimal(0),
    )
    machine.position = position
    machine._bar_minutes = 30
    return position


def test_stall_ratchet_waits_then_locks_breakeven() -> None:
    machine = _machine(stall_breakeven_atr=1.5, stall_bars=3, trail_trigger_atr=99.0,
                       breakeven_trigger_atr=99.0, take_profit_atr=0.0, max_hold_hours=0.0)
    position = _position(machine)
    position.mfe_points = 2.0  # прибыль уже 2 ATR — условие «была прибыль» выполнено

    for index in range(3):
        events = machine._manage(_bar(_ms("2026-06-18", 11) + index * 60_000, "100", "100", "100.5"),
                                 IndicatorSnapshot())
        assert not events
        assert position.stop_price == Decimal("98"), "стоп не должен дёргаться до истечения stall_bars"

    machine._manage(_bar(_ms("2026-06-18", 12), "100", "100", "100.5"), IndicatorSnapshot())
    assert position.stall_done is True
    assert position.breakeven_done is True, "застой делает позицию безрисковой (освобождает слот)"
    assert position.stop_price > Decimal("100"), "стоп зафиксирован в плюсе (вход + комиссия)"


def test_stall_ratchet_resets_on_new_extreme() -> None:
    machine = _machine(stall_breakeven_atr=1.5, stall_bars=2, trail_trigger_atr=99.0,
                       breakeven_trigger_atr=99.0, take_profit_atr=0.0, max_hold_hours=0.0)
    position = _position(machine)
    position.mfe_points = 2.0

    machine._manage(_bar(_ms("2026-06-18", 11), "100", "100", "100.5"), IndicatorSnapshot())
    position.mfe_points = 3.0  # новый экстремум — счётчик застоя сбрасывается
    machine._manage(_bar(_ms("2026-06-18", 12), "100", "101", "100.5"), IndicatorSnapshot())
    assert position.stall_bars_seen == 0
    assert position.stall_done is False


def test_stall_ignored_without_prior_profit() -> None:
    machine = _machine(stall_breakeven_atr=1.5, stall_bars=2, trail_trigger_atr=99.0,
                       breakeven_trigger_atr=99.0, take_profit_atr=0.0, max_hold_hours=0.0)
    position = _position(machine)
    position.mfe_points = 0.4  # прибыли не было — ратчет не включается

    for index in range(5):
        machine._manage(_bar(_ms("2026-06-18", 11) + index * 60_000, "100", "100", "100.5"),
                        IndicatorSnapshot())
    assert position.stall_done is False
    assert position.stop_price == Decimal("98")


def test_force_close_uses_explicit_price_and_roll_reason() -> None:
    machine = _machine()
    position = _position(machine)
    event = machine.force_close(
        _bar(_ms("2026-06-18", 18), "100", "100", "98"), BreakoutExitReason.ROLL, price=Decimal("97.5")
    )
    assert event is not None
    assert event.price == Decimal("97.5")
    assert event.reason is BreakoutExitReason.ROLL
    assert machine.position is None
    # Ролл — не стоп-аут: серия убытков и кулдаун не включаются.
    assert machine._consecutive_stops == 0
    assert position.stop_price == Decimal("98")


# ---------------------------------------------------------------- лимит ГО на позицию


def test_walkforward_settings_default_excludes_foreign_indexes() -> None:
    from bot_a.backtest.walkforward import WalkForwardSettings

    settings = WalkForwardSettings()
    assert "NASD_CONT" in settings.exclude_tickers
    assert "SPYF_CONT" in settings.exclude_tickers
    assert settings.max_position_margin_pct > 0


def test_swing_account_caps_margin_per_position() -> None:
    from bot_a.backtest.swing_portfolio import SwingAccountBacktester, SwingAccountSettings

    meta = _meta("HEAVY_CONT", step="1", step_price="1", margin=12_000)
    params = BreakoutParams().overrides(take_profit_atr=0.0)
    engine = SwingAccountBacktester(
        [(meta, params)],
        SwingAccountSettings(
            capital_rub=50_000.0,
            risk_pct_per_trade=5.0,
            max_margin_utilization_pct=75.0,
            max_position_margin_pct=25.0,
            margin_safety_buffer_pct=0.0,
        ),
    )
    # Риск позволяет много контрактов, но 25 % от 37 500 ₽ лимита ГО / 12 000 ₽ = 0.
    assert engine.size(meta, stop_points=1.0, equity=50_000.0, margin_used=0.0) == 0

    engine_no_cap = SwingAccountBacktester(
        [(meta, params)],
        SwingAccountSettings(
            capital_rub=50_000.0,
            risk_pct_per_trade=5.0,
            max_margin_utilization_pct=75.0,
            max_position_margin_pct=0.0,
            margin_safety_buffer_pct=0.0,
        ),
    )
    assert engine_no_cap.size(meta, stop_points=1.0, equity=50_000.0, margin_used=0.0) == 3
