"""Тесты веб-дашборда свинг-корзины: снапшот, HTTP-эндпоинты, рендер страницы."""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from bot_a.engine.breakout_fsm import Bar, BreakoutEvent, BreakoutPosition, BreakoutSide
from bot_a.engine.breakout_fsm import BreakoutParams
from bot_a.models.domain import InstrumentMetadata, MarketKind
from bot_a.services.swing_dashboard import SwingDashboard
from bot_a.services.swing_runner import SwingBasketRunner, SwingBasketSettings
from bot_a.utils.time_provider import MSK


def _meta(ticker: str = "TEST_CONT") -> InstrumentMetadata:
    return InstrumentMetadata(
        instrument_uid=ticker,
        ticker=ticker,
        class_code="SPBFUT",
        market=MarketKind.FUTURES,
        min_price_increment=Decimal("1"),
        min_price_increment_amount=Decimal("1"),
        lot_size=1,
        initial_margin_buy=Decimal("1000"),
        initial_margin_sell=Decimal("1000"),
        name=f"{ticker} name",
    )


def _ms(hour: int = 10) -> int:
    return int(datetime(2026, 6, 18, hour, tzinfo=MSK).timestamp() * 1000)


def _runner(bars: int = 5) -> SwingBasketRunner:
    params = BreakoutParams().overrides(take_profit_atr=0.0)
    runner = SwingBasketRunner(
        [(_meta(), params)],
        SwingBasketSettings(
            capital_rub=50_000.0,
            bar_minutes=60,
            dashboard_bars=400,
            max_margin_utilization_pct=75.0,
        ),
    )
    history = [
        Bar(
            time_ms=_ms(9) + index * 3_600_000,
            open=Decimal(100 + index),
            high=Decimal(101 + index),
            low=Decimal(99 + index),
            close=Decimal(100 + index),
            volume=10,
        )
        for index in range(bars)
    ]
    runner.warmup({"TEST_CONT": history})
    return runner


def test_snapshot_has_portfolio_and_bars() -> None:
    runner = _runner()
    snapshot = runner.snapshot()
    assert snapshot["barMinutes"] == 60
    assert snapshot["portfolio"]["capital"] == 50_000.0
    assert snapshot["portfolio"]["marginLimitRub"] == 37_500.0
    instruments = snapshot["instruments"]
    assert len(instruments) == 1
    inst = instruments[0]
    assert inst["ticker"] == "TEST_CONT"
    assert len(inst["bars"]) == 5
    assert inst["bars"][-1][4] == 104.0
    assert inst["position"] is None


def test_snapshot_exposes_stop_and_unrealized() -> None:
    runner = _runner()
    machine = runner._machines["TEST_CONT"]
    machine.position = BreakoutPosition(
        side=BreakoutSide.LONG,
        quantity=2,
        entry_price=Decimal("100"),
        entry_ms=_ms(10),
        atr_points=2.0,
        stop_price=Decimal("97"),
        initial_stop=Decimal("97"),
        fee_round_trip_rub=Decimal("1"),
        max_price=Decimal("104"),
        target_price=Decimal("108"),
    )
    runner._open(
        "TEST_CONT",
        BreakoutEvent(kind="entry", time_ms=_ms(10), side=BreakoutSide.LONG, price=Decimal("100")),
    )
    snapshot = runner.snapshot()
    position = snapshot["instruments"][0]["position"]
    assert position is not None
    assert position["side"] == "Long"
    assert 0 < position["stop"] < position["entry"]  # стоп ниже входа для Long
    assert position["target"] == 108.0
    assert position["qty"] >= 1
    assert snapshot["portfolio"]["openPositions"] == 1


def test_dashboard_serves_page_and_snapshot() -> None:
    runner = _runner()
    dashboard = SwingDashboard(runner, port=0).start()
    try:
        base = f"http://127.0.0.1:{dashboard.port}"
        with urllib.request.urlopen(f"{base}/", timeout=5) as response:
            page = response.read().decode("utf-8")
        assert response.status == 200
        assert "Bot A" in page
        assert "TEST_CONT" not in page  # тикеры приходят из API, а не зашиты в HTML
        assert "/api/snapshot" in page

        with urllib.request.urlopen(f"{base}/api/snapshot", timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        assert data["instruments"][0]["ticker"] == "TEST_CONT"

        with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
            assert response.read() == b"ok"
    finally:
        dashboard.stop()


def test_dashboard_unknown_path_returns_404() -> None:
    dashboard = SwingDashboard(_runner(), port=0).start()
    try:
        url = f"http://127.0.0.1:{dashboard.port}/nope"
        try:
            urllib.request.urlopen(url, timeout=5)
            raise AssertionError("ожидался 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        dashboard.stop()


def test_dashboard_stop_is_idempotent() -> None:
    dashboard = SwingDashboard(_runner(), port=0)
    dashboard.stop()  # до старта — не падает
    dashboard.start()
    dashboard.stop()
    dashboard.stop()
