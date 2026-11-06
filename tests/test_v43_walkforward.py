"""Тесты v4.3: прогревочное окно, освобождение слота на breakeven, гибридное ядро
и ранговый conviction."""

from __future__ import annotations

import pytest

from bot_a.backtest.walkforward import (
    Signal,
    WalkForwardSettings,
    build_signals,
    run_walkforward,
    select_basket,
)
from bot_a.backtest.swing_portfolio import SwingAccountSettings
from bot_a.engine.breakout_fsm import BreakoutSide
from bot_a.services.front_contracts import underlying_family
from bot_a.utils.quotation import dec
from tests.helpers import make_meta
from tests.test_v42_walkforward import DAY, _ms, _signal
from tests.test_v41_multiregime import trade_params, _trending_cycles


# --------------------------------------------------------------- v4.3: параметры
def test_new_settings_defaults() -> None:
    settings = WalkForwardSettings()
    assert settings.free_slot_on_breakeven is True
    assert settings.max_total_open_positions == 8
    assert settings.warmup_days == 0
    assert settings.conviction_leader_boost == pytest.approx(1.45)
    portfolio = SwingAccountSettings()
    assert portfolio.free_slot_on_breakeven is False  # бэктест по умолчанию как раньше
    assert portfolio.max_total_open_positions == 8


# --------------------------------------------------------------- слоты на breakeven
def _slot_streams(free_delay_ms: int):
    """A держит слот (или уже безрисковый), B хочет войти в тот же срез.

    Стоп в 40 пунктов держит размер позиции маленьким, чтобы лимит ГО не мешал
    проверять именно лимит риск-слотов.
    """
    start_ms = _ms("2026-05-01")
    entry_a = start_ms + 46 * DAY
    stop = 40.0
    setup = {
        ("AAA", "A"): [
            _signal("AAA", start_ms + 2 * DAY, start_ms + 3 * DAY, entry="100", exit_price="102",
                    stop_points=stop),
            _signal("AAA", start_ms + 16 * DAY, start_ms + 17 * DAY, entry="100", exit_price="102",
                    stop_points=stop),
            _signal("AAA", entry_a, entry_a + 20 * DAY, entry="100", exit_price="103", stop_points=stop),
        ],
        ("BBB", "A"): [
            _signal("BBB", start_ms + 2 * DAY, start_ms + 3 * DAY, entry="100", exit_price="102",
                    stop_points=stop),
            _signal("BBB", start_ms + 16 * DAY, start_ms + 17 * DAY, entry="100", exit_price="102",
                    stop_points=stop),
            _signal("BBB", entry_a + 60_000, entry_a + 21 * DAY, entry="100", exit_price="103",
                    stop_points=stop),
        ],
    }
    setup[("AAA", "A")][-1].free_ms = entry_a + free_delay_ms
    return setup, start_ms


def test_free_slot_on_breakeven_allows_new_entry() -> None:
    metas = {
        "AAA": make_meta("AAA", price_step="1", step_amount="1", margin="900"),
        "BBB": make_meta("BBB", price_step="1", step_amount="1", margin="900"),
    }
    streams, start_ms = _slot_streams(free_delay_ms=60_000)  # AAA стала безрисковой через час
    settings = WalkForwardSettings(
        capital_rub=50_000, risk_pct_per_trade=0.7, step_days=30, lookback_days=45,
        top=2, min_pf=1.0, min_trades=1, max_concurrent_positions=1,
        free_slot_on_breakeven=True,
    )
    result = run_walkforward(streams, metas, start_ms, _ms("2026-09-01"), settings)
    tickers = {trade.ticker for trade in result.trades}
    assert "AAA" in tickers and "BBB" in tickers, "безрисковая позиция не должна блокировать слот"


def test_without_free_slot_new_entry_is_blocked() -> None:
    metas = {
        "AAA": make_meta("AAA", price_step="1", step_amount="1", margin="900"),
        "BBB": make_meta("BBB", price_step="1", step_amount="1", margin="900"),
    }
    streams, start_ms = _slot_streams(free_delay_ms=60_000)
    settings = WalkForwardSettings(
        capital_rub=50_000, risk_pct_per_trade=0.7, step_days=30, lookback_days=45,
        top=2, min_pf=1.0, min_trades=1, max_concurrent_positions=1,
        free_slot_on_breakeven=False,
    )
    result = run_walkforward(streams, metas, start_ms, _ms("2026-09-01"), settings)
    assert result.rejected_by_slots > 0


def test_total_open_positions_cap_is_enforced() -> None:
    metas = {
        "AAA": make_meta("AAA", price_step="1", step_amount="1", margin="900"),
        "BBB": make_meta("BBB", price_step="1", step_amount="1", margin="900"),
    }
    streams, start_ms = _slot_streams(free_delay_ms=60_000)
    settings = WalkForwardSettings(
        capital_rub=50_000, risk_pct_per_trade=0.7, step_days=30, lookback_days=45,
        top=2, min_pf=1.0, min_trades=1, max_concurrent_positions=10,
        free_slot_on_breakeven=True, max_total_open_positions=1,
    )
    result = run_walkforward(streams, metas, start_ms, _ms("2026-09-01"), settings)
    assert result.rejected_by_slots > 0


# --------------------------------------------------------------- гибридное ядро
def _pair_streams(ticker: str, entry_edge: str, exit_offset_day: int = 1):
    at = _ms("2026-09-01")
    start = at - 60 * DAY
    middle = (start + at) // 2
    return [
        _signal(ticker, start + DAY, start + 2 * DAY, exit_price=entry_edge),
        _signal(ticker, middle + DAY, middle + 2 * DAY, exit_price=entry_edge),
    ]


def test_core_anchor_allowed_below_min_pf() -> None:
    at = _ms("2026-09-01")
    metas = {
        "CNYRUBF": make_meta("CNYRUBF", price_step="1", step_amount="1"),
        "BRM_CONT": make_meta("BRM_CONT", price_step="1", step_amount="1"),
    }
    # Умеренный край: PF около 1.2 (< min_pf 1.30, но >= core_min_pf 1.15).
    # PF 1.2 в обеих половинах окна: +6/−5 и +6/−5.
    streams = {
        ("CNYRUBF", "A"): [
            _signal("CNYRUBF", at - 50 * DAY, at - 49 * DAY, entry="100", exit_price="106"),
            _signal("CNYRUBF", at - 47 * DAY, at - 46 * DAY, entry="100", exit_price="95"),
            _signal("CNYRUBF", at - 20 * DAY, at - 19 * DAY, entry="100", exit_price="106"),
            _signal("CNYRUBF", at - 17 * DAY, at - 16 * DAY, entry="100", exit_price="95"),
        ]
    }
    without = select_basket(streams, metas, at, lookback_days=60, min_pf=1.30, min_trades=2)
    with_core = select_basket(
        streams, metas, at, lookback_days=60, min_pf=1.30, min_trades=2,
        core_anchors=("CNYRUBF",), core_min_pf=1.15,
    )
    assert without == []
    assert [item["ticker"] for item in with_core] == ["CNYRUBF"]


def test_satellite_min_pf_filters_weak_pairs() -> None:
    at = _ms("2026-09-01")
    metas = {"BRM_CONT": make_meta("BRM_CONT", price_step="1", step_amount="1")}
    # PF 1.2 (12/10) → проходит min_pf 1.15, но не проходит satellite_min_pf 1.45.
    streams = {
        ("BRM_CONT", "A"): [
            _signal("BRM_CONT", at - 50 * DAY, at - 49 * DAY, entry="100", exit_price="106"),
            _signal("BRM_CONT", at - 47 * DAY, at - 46 * DAY, entry="100", exit_price="95"),
            _signal("BRM_CONT", at - 20 * DAY, at - 19 * DAY, entry="100", exit_price="106"),
            _signal("BRM_CONT", at - 17 * DAY, at - 16 * DAY, entry="100", exit_price="95"),
        ]
    }
    loose = select_basket(streams, metas, at, lookback_days=60, min_pf=1.15, min_trades=2)
    strict = select_basket(
        streams, metas, at, lookback_days=60, min_pf=1.15, min_trades=2, satellite_min_pf=1.45
    )
    assert loose and not strict


def test_excluded_tickers_are_skipped() -> None:
    at = _ms("2026-09-01")
    metas = {
        "NASD_CONT": make_meta("NASD_CONT", price_step="1", step_amount="1"),
        "BRM_CONT": make_meta("BRM_CONT", price_step="1", step_amount="1"),
    }
    streams = {
        ("NASD_CONT", "A"): _pair_streams("NASD_CONT", "104"),
        ("BRM_CONT", "A"): _pair_streams("BRM_CONT", "102"),
    }
    basket = select_basket(
        streams, metas, at, lookback_days=60, min_pf=1.0, min_trades=2,
        exclude_tickers=("NASD_CONT",),
    )
    assert [item["ticker"] for item in basket] == ["BRM_CONT"]


def test_basket_sets_ranks() -> None:
    at = _ms("2026-09-01")
    metas = {
        "BRM_CONT": make_meta("BRM_CONT", price_step="1", step_amount="1"),
        "GAZR_CONT": make_meta("GAZR_CONT", price_step="1", step_amount="1"),
    }
    streams = {
        ("BRM_CONT", "A"): _pair_streams("BRM_CONT", "104"),
        ("GAZR_CONT", "A"): _pair_streams("GAZR_CONT", "102"),
    }
    basket = select_basket(streams, metas, at, lookback_days=60, min_pf=1.0, min_trades=2)
    assert [item["rank"] for item in basket] == [1, 2]
    assert basket[0]["ticker"] == "BRM_CONT"


# --------------------------------------------------------------- ранговый conviction
def test_conviction_leader_boost_increases_size() -> None:
    metas = {"BRM_CONT": make_meta("BRM_CONT", price_step="1", step_amount="1", margin="900")}
    start_ms = _ms("2026-05-01")
    series = [
        _signal("BRM_CONT", start_ms + 2 * DAY, start_ms + 3 * DAY, entry="100", exit_price="102"),
        _signal("BRM_CONT", start_ms + 16 * DAY, start_ms + 17 * DAY, entry="100", exit_price="102"),
        _signal("BRM_CONT", start_ms + 46 * DAY, start_ms + 48 * DAY, entry="100", exit_price="104"),
    ]
    for item in series:
        item.conviction = 1.2
    streams = {("BRM_CONT", "A"): series}

    def quantity(conviction_sizing: bool) -> int:
        settings = WalkForwardSettings(
            capital_rub=50_000, risk_pct_per_trade=0.7, step_days=30, lookback_days=45,
            top=1, min_pf=1.0, min_trades=1, max_concurrent_positions=4,
            conviction_sizing=conviction_sizing, conviction_leader_boost=1.45,
        )
        result = run_walkforward(streams, metas, start_ms, _ms("2026-09-01"), settings)
        return max((trade.quantity for trade in result.trades), default=0)

    plain = quantity(False)
    boosted = quantity(True)
    assert plain > 0 and boosted >= plain


# --------------------------------------------------------------- прогревочные данные
def test_build_signals_marks_free_ms() -> None:
    meta = make_meta("AAA", price_step="0.01", step_amount="0.01")
    params = trade_params(min_efficiency_ratio=0.0, close_in_range=1.0)
    candles = _trending_cycles(cycles=10)
    signals = build_signals(meta, params, candles, "A")
    assert signals
    # free_ms либо 0 (до breakeven не дошло), либо внутри жизни сделки.
    for signal in signals:
        assert signal.free_ms == 0 or signal.entry_ms <= signal.free_ms <= signal.exit_ms


def test_prewarm_history_is_available() -> None:
    """v4.3: в кэше есть свечи с 05.03.2026 — иначе трейлинг-окно не готово к 25.04."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from breakout_scan import load_candles

    candles = load_candles("BRM_CONT", "2026-03-05", "2026-04-24", 30)
    assert len(candles) > 100, "прогрев 05.03–24.04 не скачан"
    first = min(candle.time_ms for candle in candles)
    assert first <= _ms("2026-03-31")
