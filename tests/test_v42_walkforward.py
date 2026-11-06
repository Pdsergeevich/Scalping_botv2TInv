"""Тесты v4.2: роллинг walk-forward, conviction-сайзинг и метрики по фактической equity."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal

import pytest

from bot_a.backtest.walkforward import (
    Signal,
    WalkForwardSettings,
    _window_stats,
    build_signals,
    engine_params,
    run_walkforward,
    select_basket,
    update_appsettings,
)
from bot_a.backtest.breakout_backtest import BreakoutTrade
from bot_a.backtest.swing_portfolio import (
    SwingAccountBacktester,
    SwingAccountSettings,
    equity_metrics,
)
from bot_a.engine.breakout_fsm import BreakoutExitReason, BreakoutParams, BreakoutSide
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import MSK
from tests.helpers import make_meta
from tests.test_v32_swing import minute_ms
from tests.test_v41_multiregime import _trending_cycles, trade_params


DAY = 86_400_000


def _ms(day: str, hour: int = 12) -> int:
    return int(datetime.fromisoformat(day).replace(hour=hour, tzinfo=MSK).timestamp() * 1000)


def _signal(
    ticker: str,
    entry_ms: int,
    exit_ms: int,
    *,
    entry: str = "100",
    exit_price: str = "101",
    side: BreakoutSide = BreakoutSide.LONG,
    stop_points: float = 1.0,
    conviction: float = 1.0,
    engine: str = "A",
) -> Signal:
    from bot_a.services.front_contracts import underlying_family

    return Signal(
        ticker=ticker,
        engine=engine,
        entry_ms=entry_ms,
        exit_ms=exit_ms,
        side=side,
        entry_price=dec(entry),
        exit_price=dec(exit_price),
        stop_points=stop_points,
        conviction=conviction,
        family=underlying_family(ticker),
    )


def _trade(ticker: str, net: float, entry_ms: int = 0, exit_ms: int = 1, quantity: int = 1):
    return BreakoutTrade(
        ticker=ticker,
        side=BreakoutSide.LONG,
        quantity=quantity,
        entry_ms=entry_ms,
        exit_ms=exit_ms,
        entry_price=dec("100"),
        exit_price=dec("100"),
        gross_rub=net,
        fees_rub=0.0,
        net_rub=net,
        reason=BreakoutExitReason.HARD_STOP,
        atr_points=0.0,
        mfe_points=0.0,
        hold_minutes=1.0,
    )


# --------------------------------------------------------------- наборы движков
def test_engine_params_flags() -> None:
    base = BreakoutParams()
    assert engine_params(base, "A").pullback_enabled is False
    assert engine_params(base, "A").fade_enabled is False
    assert engine_params(base, "B").pullback_enabled is True
    assert engine_params(base, "B").fade_enabled is False
    assert engine_params(base, "C").fade_enabled is True
    ab = engine_params(base, "AB")
    assert ab.pullback_enabled and not ab.fade_enabled and ab.regime_mode == "multi"
    abc = engine_params(base, "ABC")
    assert abc.pullback_enabled and abc.fade_enabled


def test_build_signals_are_self_contained() -> None:
    meta = make_meta("AAA", price_step="0.01", step_amount="0.01")
    candles = _trending_cycles()
    signals = build_signals(meta, trade_params(min_efficiency_ratio=0.0, close_in_range=1.0), candles, "A")
    assert signals, "синтетика должна давать сделки"
    for signal in signals:
        assert signal.entry_ms < signal.exit_ms
        assert signal.stop_points > 0
        assert 0.75 <= signal.conviction <= 1.35
        assert signal.entry_price > 0 and signal.exit_price > 0


# --------------------------------------------------------------- окно отбора
def test_window_stats_counts_only_closed_trades() -> None:
    meta = make_meta("AAA", price_step="0.01", step_amount="0.01")
    start = _ms("2026-06-01")
    end = _ms("2026-07-01")
    inside = _signal("AAA", start + 5 * DAY, start + 10 * DAY)
    # Сделка ещё не закрыта на конец окна — не должна учитываться (нет lookahead).
    open_at_end = _signal("AAA", start + 20 * DAY, end + 5 * DAY)
    stats = _window_stats([inside, open_at_end], start, end, meta)
    assert stats["trades"] == 1


def test_window_stats_is_net_of_fees() -> None:
    # Шаг 1 пункт = 1 ₽ нотионала 100 ₽: комиссия 0.025 %×2 ≈ 0.05 ₽, край 0.01 ₽ → минус.
    meta = make_meta("AAA", price_step="1", step_amount="1")
    start = _ms("2026-06-01")
    tiny = _signal("AAA", start + DAY, start + 2 * DAY, entry="100", exit_price="100.01")
    stats = _window_stats([tiny], start, _ms("2026-07-01"), meta)
    assert stats["net"] < 0


def test_select_basket_requires_both_halves_positive() -> None:
    meta = make_meta("AAA", price_step="1", step_amount="1")
    at_ms = _ms("2026-09-01")
    start_window = at_ms - 60 * DAY
    middle = (start_window + at_ms) // 2
    # Первая половина убыточна, вторая прибыльна → кандидат отбрасывается.
    bad = [
        _signal("AAA", start_window + DAY, start_window + 2 * DAY, exit_price="99"),
        _signal("AAA", start_window + 40 * DAY, start_window + 14 * DAY * 3, exit_price="103"),
    ]
    good = [
        _signal("BBB", start_window + DAY, start_window + 2 * DAY, exit_price="102"),
        _signal("BBB", middle + DAY, middle + 2 * DAY, exit_price="102"),
    ]
    metas = {
        "AAA": make_meta("AAA", price_step="1", step_amount="1"),
        "BBB": make_meta("BBB", price_step="1", step_amount="1"),
    }
    basket = select_basket(
        {("AAA", "A"): bad, ("BBB", "A"): good}, metas, at_ms,
        lookback_days=60, min_pf=1.0, min_trades=2,
    )
    assert [item["ticker"] for item in basket] == ["BBB"]


def test_select_basket_deduplicates_families_and_caps_top() -> None:
    metas = {
        "CNYRUBF": make_meta("CNYRUBF", price_step="1", step_amount="1"),
        "CNY_CONT": make_meta("CNY_CONT", price_step="1", step_amount="1"),
        "BRM_CONT": make_meta("BRM_CONT", price_step="1", step_amount="1"),
    }
    at_ms = _ms("2026-09-01")
    start_window = at_ms - 60 * DAY
    middle = (start_window + at_ms) // 2
    streams = {}
    for ticker, edge in (("CNYRUBF", "102"), ("CNY_CONT", "104"), ("BRM_CONT", "101")):
        streams[(ticker, "A")] = [
            _signal(ticker, start_window + DAY, start_window + 2 * DAY, exit_price=edge),
            _signal(ticker, middle + DAY, middle + 2 * DAY, exit_price=edge),
        ]
    basket = select_basket(streams, metas, at_ms, lookback_days=60, min_pf=1.0, min_trades=2, top=5)
    families = [item["family"] for item in basket]
    assert len(families) == len(set(families))  # никаких дублей активов
    assert "CNY_CONT" in [item["ticker"] for item in basket]  # остался лучший по Net
    capped = select_basket(streams, metas, at_ms, lookback_days=60, min_pf=1.0, min_trades=2, top=1)
    assert len(capped) == 1


def test_select_basket_respects_min_trades_and_pf() -> None:
    metas = {"AAA": make_meta("AAA", price_step="1", step_amount="1")}
    at_ms = _ms("2026-09-01")
    start_window = at_ms - 60 * DAY
    streams = {
        ("AAA", "A"): [
            _signal("AAA", start_window + DAY, start_window + 2 * DAY, exit_price="100.5"),
            _signal("AAA", start_window + 3 * DAY, start_window + 4 * DAY, exit_price="99.5"),
        ]
    }
    # 1 сделка в каждой половине, но PF ≈ 1.0 < 1.30 и сделок < 4.
    assert select_basket(streams, metas, at_ms, lookback_days=60, min_pf=1.30, min_trades=4) == []


# --------------------------------------------------------------- исполнение
def test_run_walkforward_records_rotation_and_true_metrics() -> None:
    metas = {
        "AAA": make_meta("AAA", price_step="1", step_amount="1", margin="1000"),
        "BBB": make_meta("BBB", price_step="1", step_amount="1", margin="1000"),
    }
    start_ms = _ms("2026-05-01")
    end_ms = _ms("2026-09-01")
    streams = {}
    for index, ticker in enumerate(("AAA", "BBB")):
        series = []
        for week in range(14):
            entry = start_ms + (week * 7 + 1) * DAY + index * 60_000
            series.append(
                _signal(ticker, entry, entry + 2 * DAY, entry="100", exit_price="101", stop_points=1.0)
            )
        streams[(ticker, "A")] = series
    settings = WalkForwardSettings(
        capital_rub=50_000,
        risk_pct_per_trade=0.7,
        step_days=14,
        lookback_days=45,
        top=2,
        min_pf=1.0,
        min_trades=2,
        conviction_sizing=False,
    )
    result = run_walkforward(streams, metas, start_ms, end_ms, settings)
    assert result.baskets, "ротация должна зафиксировать хотя бы одну корзину"
    assert result.trades, "сигналы в срезе должны исполняться"
    stats = result.metrics(50_000)
    assert stats["trades"] == len(result.trades)
    expected = sum(trade.net_rub for trade in result.trades)
    assert stats["netRub"] == pytest.approx(expected, abs=0.01)
    assert stats["monthly"]


def test_run_walkforward_respects_slots() -> None:
    metas = {
        f"T{index}": make_meta(f"T{index}", price_step="1", step_amount="1", margin="900")
        for index in range(4)
    }
    start_ms = _ms("2026-05-01")
    end_ms = _ms("2026-09-01")
    # Входы — уже после первой ротации (первая корзина собирается на 45-й день),
    # по 4 сигнала почти одновременно: при лимите 1 слот три должны быть отклонены.
    entry_base = start_ms + 60 * DAY
    history_base = start_ms + 2 * DAY
    streams = {}
    for index, ticker in enumerate(metas):
        # История для отбора (закрытые сделки внутри трейлинг-окна) …
        history = [
            _signal(ticker, history_base + step * DAY, history_base + (step + 1) * DAY,
                    entry="100", exit_price="102", stop_points=1.0)
            for step in (0, 14, 28)
        ]
        # … и «боевые» входы после первой ротации, почти одновременно.
        entry = entry_base + index * 60_000
        history.append(
            _signal(ticker, entry, entry + 2 * DAY, entry="100", exit_price="102", stop_points=1.0)
        )
        streams[(ticker, "A")] = history
    settings = WalkForwardSettings(
        capital_rub=50_000, risk_pct_per_trade=0.7, step_days=30, lookback_days=45,
        top=4, min_pf=1.0, min_trades=1, max_concurrent_positions=1,
    )
    result = run_walkforward(streams, metas, start_ms, end_ms, settings)
    assert result.trades, "первый сигнал должен открыться"
    assert result.rejected_by_slots > 0, "остальные обязаны упасть в лимит слотов"


# --------------------------------------------------------------- метрики equity
def test_equity_metrics_drawdown_and_pf() -> None:
    curve = [
        (_ms("2026-05-04"), 50_000.0),
        (_ms("2026-05-05"), 52_000.0),
        (_ms("2026-05-06"), 47_000.0),  # просадка 5 000 ₽ (9.6 %)
        (_ms("2026-05-07"), 53_000.0),
    ]
    trades = [_trade("AAA", 3_000.0), _trade("AAA", -2_000.0)]
    metrics = equity_metrics(curve, trades, 50_000)
    assert metrics["netRub"] == pytest.approx(3_000.0)
    assert metrics["maxDrawdownRub"] == pytest.approx(5_000.0)
    assert metrics["maxDrawdownPct"] == pytest.approx(5_000 / 52_000 * 100, abs=0.01)
    assert metrics["profitFactor"] == pytest.approx(1.5)
    assert metrics["trades"] == 2
    monthly = metrics["monthly"]
    assert monthly.get("2026-05") == pytest.approx(3_000.0)


def test_equity_metrics_empty_curve() -> None:
    metrics = equity_metrics([], [], 50_000)
    assert metrics["netRub"] == 0.0
    assert metrics["profitFactor"] is None


def test_backtester_metrics_match_equity_curve() -> None:
    params = trade_params(min_efficiency_ratio=0.0, close_in_range=1.0)
    engine = SwingAccountBacktester(
        [(make_meta("AAA", price_step="0.01", step_amount="0.01", margin="1000"), params)],
        SwingAccountSettings(capital_rub=50_000, risk_pct_per_trade=1.0),
    )
    result = engine.run({"AAA": _trending_cycles()})
    stats = result.metrics(50_000)
    assert stats["trades"] == len(result.trades)
    if result.equity_curve:
        assert stats["netRub"] == pytest.approx(result.equity_curve[-1][1] - 50_000, abs=0.01)


# --------------------------------------------------------------- conviction
def test_conviction_sizing_disabled_by_default() -> None:
    assert SwingAccountSettings().conviction_sizing is False


def test_conviction_multiplier_bounds_and_tiers() -> None:
    engine = SwingAccountBacktester(
        [(make_meta("AAA"), trade_params())],
        SwingAccountSettings(capital_rub=50_000, conviction_sizing=True),
    )
    engine._conviction = 1.0  # noqa: SLF001
    assert engine._conviction_multiplier("AAA") == pytest.approx(1.0)  # noqa: SLF001
    engine._conviction = 1.35  # noqa: SLF001
    engine._ticker_pf["AAA"] = 2.0  # noqa: SLF001
    assert engine._conviction_multiplier("AAA") == pytest.approx(1.35)  # noqa: SLF001 — потолок
    engine._conviction = 0.75  # noqa: SLF001
    engine._ticker_pf["AAA"] = 0.9  # noqa: SLF001
    assert engine._conviction_multiplier("AAA") == pytest.approx(0.75)  # noqa: SLF001 — пол


def test_fsm_conviction_score_range() -> None:
    from bot_a.engine.breakout_fsm import Bar
    from bot_a.engine.multiregime_fsm import MultiRegimeStateMachine

    machine = MultiRegimeStateMachine(
        make_meta("AAA", price_step="0.01", step_amount="0.01"), trade_params()
    )
    for candle in _trending_cycles()[:80]:
        machine.on_bar(
            Bar(
                time_ms=candle.time_ms,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=int(candle.volume),
            )
        )
    bar = Bar(
        time_ms=candle.time_ms,
        open=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        volume=int(candle.volume),
    )
    score = machine._conviction_score(bar, "trend")  # noqa: SLF001
    assert 0.75 <= score <= 1.35


# --------------------------------------------------------------- обновление конфига
def test_update_appsettings_writes_basket_and_engines(tmp_path) -> None:
    path = tmp_path / "appsettings.json"
    path.write_text(
        json.dumps({"strategyMode": "breakout", "breakoutPresets": {"CNYRUBF": {"donchianBars": 20}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    config = update_appsettings(
        path,
        {"CNYRUBF": "B", "BRM_CONT": "ABC"},
        {"riskPctPerTrade": 0.7, "maxConcurrentPositions": 4},
    )
    assert config["swingBasket"] == ["CNYRUBF", "BRM_CONT"]
    assert config["swingPortfolio"]["riskPctPerTrade"] == 0.7
    cny = config["breakoutPresets"]["CNYRUBF"]
    assert cny["pullbackEnabled"] is True and cny["fadeEnabled"] is False
    assert cny["donchianBars"] == 20  # прочие настройки пресета сохранены
    brm = config["breakoutPresets"]["BRM_CONT"]
    assert brm["pullbackEnabled"] is True and brm["fadeEnabled"] is True


def test_live_appsettings_have_walk_forward_basket() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "appsettings.json").read_text(encoding="utf-8"))
    basket = config.get("swingBasket") or []
    assert basket, "swingBasket должен быть заполнен для live/paper"
    assert config["swingPortfolio"]["riskPctPerTrade"] > 0
    for ticker in basket:
        assert ticker in config["breakoutPresets"], f"нет пресета для {ticker}"
