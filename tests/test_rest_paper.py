"""Тесты v2.4.1 live-обвязки: REST paper-сервис и загрузчик токена."""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from pathlib import Path

from bot_a.broker.tinkoff_rest_client import TInvestRestClient
from bot_a.models.domain import OrderBookLevel, OrderBookSnapshot, TradePrint, Side
from bot_a.services.rest_paper import RestPaperService
from bot_a.utils.secrets import mask, read_token, token_source
from tests.helpers import make_config, make_meta, make_preset

from tests.test_engine import _params


# ------------------------------------------------------------------ токен
def test_token_from_local_secret_file(tmp_path):
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "tinvest_token.txt").write_text("# комментарий\nt.test-token-123\n", encoding="utf-8")
    assert read_token(env_var="BOT_A_TEST_TOKEN_MISSING", base_dir=tmp_path) == "t.test-token-123"
    assert "локальный файл" in token_source(env_var="BOT_A_TEST_TOKEN_MISSING", base_dir=tmp_path)


def test_token_priority_env_over_file(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "tinvest_token.txt").write_text("t.from-file\n", encoding="utf-8")
    monkeypatch.setenv("BOT_A_TEST_TOKEN_PRIORITY", "t.from-env")
    assert read_token(env_var="BOT_A_TEST_TOKEN_PRIORITY", base_dir=tmp_path) == "t.from-env"
    # явный аргумент сильнее окружения
    assert read_token("t.explicit", env_var="BOT_A_TEST_TOKEN_PRIORITY", base_dir=tmp_path) == "t.explicit"


def test_token_file_env_var(tmp_path, monkeypatch):
    custom = tmp_path / "custom.txt"
    custom.write_text("t.custom-file\n", encoding="utf-8")
    monkeypatch.setenv("TINVEST_TOKEN_FILE", str(custom))
    assert read_token(env_var="BOT_A_TEST_TOKEN_MISSING2", base_dir=tmp_path) == "t.custom-file"


def test_mask_never_reveals_full_token():
    assert mask("") == "не задан"
    assert mask("short") == "*****"
    text = mask("t.abcdefghijklmnop")
    assert "abcdefghij" not in text
    assert text.startswith("t.ab")


def test_config_resolve_token_uses_base_dir(tmp_path):
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "tinvest_token.txt").write_text("t.config-token\n", encoding="utf-8")
    config = make_config(tmp_path)
    from dataclasses import replace

    config = replace(config, t_invest=replace(config.t_invest, token="", token_env_var="BOT_A_NO_SUCH"))
    assert config.t_invest.resolve_token(config.base_dir) == "t.config-token"
    assert "локальный файл" in config.t_invest.token_source(config.base_dir)


# ------------------------------------------------------- REST paper pipeline
class _FakeRestClient:
    """Мок REST-клиента: отдаёт синтетический стакан с заданным дрейфом."""

    def __init__(self, books: int = 40, drift: str = "0.02"):
        self._books = books
        self._drift = Decimal(drift)
        self._index = 0
        self._base_ms = 1_700_000_000_000

    async def get_order_book(self, instrument_uid: str, depth: int = 20):
        if self._index >= self._books:
            return None
        step = Decimal("0.01")
        shift = self._drift * Decimal(self._index)
        base = Decimal("100.00") if instrument_uid == "uid-AAA" else Decimal("200.00")
        mid = base + shift
        bids = tuple(
            OrderBookLevel(price=mid - step * (i + 1), quantity=100) for i in range(5)
        )
        asks = tuple(
            OrderBookLevel(price=mid + step * (i + 1), quantity=100) for i in range(5)
        )
        time_ms = self._base_ms + self._index * 200
        if instrument_uid == "uid-BBB":
            self._index += 1  # пара считается завершённой после второй ноги
        return OrderBookSnapshot(
            instrument_uid=instrument_uid, time_ms=time_ms, bids=bids, asks=asks,
            last_price=mid,
        )

    async def get_last_trades(self, instrument_uid: str, date_from=None, date_to=None):
        return []


def test_rest_paper_service_runs_and_collects_metrics(tmp_path):
    """Сервис кормит ядро стаканами, пишет трассу и корректно завершается."""

    class _Trace:
        def __init__(self):
            self.points = []

        def record(self, metrics, state):
            if metrics is not None:
                self.points.append((metrics.zscore, state))

    async def scenario():
        leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
        leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
        preset = make_preset(leg1, leg2)
        config = make_config(tmp_path, preset, _params(stat_sample_interval_seconds=5.0))
        client = _FakeRestClient(books=30)
        service = RestPaperService(
            config, preset, client, interval_ms=50, depth=5
        )
        trace = _Trace()
        snapshot, trades, warmup = await service.run(duration_seconds=2.0, trace=trace, warmup_days=0)
        assert snapshot["books"] >= 10, snapshot
        assert snapshot["sessionSeconds"] >= 0
        assert warmup == {}
        assert isinstance(trades, list)
        assert len(trace.points) > 0
        assert "counters" in snapshot

    asyncio.run(scenario())


def test_rest_paper_respects_stop_event(tmp_path):
    """stop_event останавливает сессию без ожидания duration."""

    async def scenario():
        leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
        leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
        preset = make_preset(leg1, leg2)
        config = make_config(tmp_path, preset, _params(stat_sample_interval_seconds=5.0))
        service = RestPaperService(config, preset, _FakeRestClient(books=10_000), interval_ms=50, depth=5)
        stop = asyncio.Event()

        async def stopper():
            await asyncio.sleep(0.3)
            stop.set()

        asyncio.create_task(stopper())
        snapshot, trades, _ = await service.run(
            duration_seconds=None, stop_event=stop, warmup_days=0
        )
        assert snapshot["sessionSeconds"] < 5.0, snapshot["sessionSeconds"]

    asyncio.run(scenario())


def test_rest_client_timeout_becomes_rest_error():
    """Таймаут сокета не должен убивать сессию: клиент превращает его в TInvestRestError."""
    import socket

    import pytest

    from bot_a.broker.tinkoff_rest_client import TInvestRestError

    def transport(url, payload, headers):
        raise socket.timeout("read timed out")

    client = TInvestRestClient("t.test", transport=transport)
    with pytest.raises(TInvestRestError) as exc:
        client._post_sync("/x", {})  # noqa: SLF001
    assert "Таймаут" in str(exc.value)


def test_rest_client_connection_error_becomes_rest_error():
    import pytest

    from bot_a.broker.tinkoff_rest_client import TInvestRestError

    def transport(url, payload, headers):
        raise ConnectionResetError("connection reset")

    client = TInvestRestClient("t.test", transport=transport)
    with pytest.raises(TInvestRestError):
        client._post_sync("/x", {})  # noqa: SLF001


def test_rest_client_rejects_bom_token():
    """Токен с BOM не должен уходить в HTTP-заголовок (ломал Authorization)."""
    import pytest

    from bot_a.broker.tinkoff_rest_client import TInvestRestError

    with pytest.raises(TInvestRestError):
        TInvestRestClient("\ufeffимя-с-юникодом")


# ------------------------------------------------------- preflight через REST
class _FakePreflightClient:
    """Мок REST-клиента для preflight: счёт, позиции, метаданные, статус."""

    def __init__(self, free_cash: str = "300000"):
        self._free = Decimal(free_cash)
        self.tickers = ("AAA", "BBB")

    async def get_accounts(self):
        return [{"id": "acc-1", "name": "Тестовый", "accessLevel": "full_access"}]

    async def resolve_instrument(self, ticker, class_code, market):
        return make_meta(ticker, uid=f"uid-{ticker}", price_step="0.01", step_amount="0.01")

    async def get_trading_status(self, instrument_uid):
        return {"tradingStatus": "NORMAL_TRADING", "marketOrderAvailable": True}

    async def get_positions(self, account_id):
        return {"money": [], "blocked": []}

    async def free_cash_rub(self, account_id):
        return self._free


def test_preflight_rest_reports_ready(tmp_path):
    """Preflight через REST даёт отчёт с счётом, метаданными и ГО."""

    async def scenario():
        from bot_a.services.preflight import run_preflight_rest

        leg1 = make_meta("AAA", uid="uid-AAA")
        leg2 = make_meta("BBB", uid="uid-BBB")
        preset = make_preset(leg1, leg2)
        config = make_config(tmp_path, preset)
        report = await run_preflight_rest(config, preset, _FakePreflightClient())
        names = {check.name for check in report.checks}
        assert "Токен T-Invest" in names
        assert "Доступ к счетам" in names
        assert "Свободные средства" in names
        assert "Метаданные Leg1 (AAA)" in names
        assert "Каденция статистики" in names
        assert report.account_id == "acc-1"
        assert report.free_cash_rub == Decimal("300000")
        assert report.leg1 is not None and report.leg1.instrument_uid == "uid-AAA"
        text = report.render()
        assert "Preflight:" in text and "Свободные RUB" in text

    asyncio.run(scenario())


def test_preflight_rest_handles_no_accounts(tmp_path):
    """Без счетов отчёт содержит FAIL и не падает."""

    class _NoAccounts(_FakePreflightClient):
        async def get_accounts(self):
            return []

    async def scenario():
        from bot_a.services.preflight import run_preflight_rest

        leg1 = make_meta("AAA", uid="uid-AAA")
        leg2 = make_meta("BBB", uid="uid-BBB")
        preset = make_preset(leg1, leg2)
        config = make_config(tmp_path, preset)
        report = await run_preflight_rest(config, preset, _NoAccounts())
        assert not report.ok
        assert any(check.name == "Выбор счёта" and not check.ok for check in report.checks)

    asyncio.run(scenario())
