"""Тесты live-дашборда (--mode watch): состояние, стакан, HTTP-слой."""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
from decimal import Decimal

from bot_a.services.live_dashboard import DashboardState, LiveWatchService
from tests.helpers import make_book, make_config, make_meta, make_preset
from tests.test_engine import _params


def test_dashboard_state_publish_and_snapshot():
    state = DashboardState()
    assert state.snapshot()["status"] == "starting"
    state.publish({"status": "live", "z": 1.5})
    assert state.snapshot()["z"] == 1.5


def test_dashboard_state_thread_safe():
    state = DashboardState()
    errors: list[Exception] = []

    def writer(index: int) -> None:
        try:
            for step in range(200):
                state.publish({"status": "live", "writer": index, "step": step})
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert state.snapshot()["status"] == "live"


def test_book_payload_shape(tmp_path):
    leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
    leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
    config = make_config(tmp_path, make_preset(leg1, leg2), _params())
    preset = config.preset()
    service = LiveWatchService(config, preset, client=None, port=0)  # type: ignore[arg-type]
    book = make_book(Decimal("100.00"), 1_700_000_000_000, uid="uid-AAA")
    payload = service._book_payload(book, levels=3)  # noqa: SLF001
    assert payload["bestBid"] is not None and payload["bestAsk"] is not None
    assert len(payload["bids"]) == 3 and len(payload["asks"]) == 3
    assert payload["asks"][0]["p"] > payload["bids"][0]["p"]


def _no_warmup(config):
    """В тестах прогрев из ISS не нужен: сеть и кэш не задействуем."""
    from dataclasses import replace

    return replace(config, execution=replace(config.execution, bootstrap_warmup_days=0))


class _FakeWatchClient:
    """Мок REST-клиента для дашборда: стакан с небольшим дрейфом."""

    def __init__(self, ticks: int = 10_000):
        self._ticks = ticks
        self._index = 0
        self._base = 1_700_000_000_000

    async def get_order_book(self, instrument_uid: str, depth: int = 20):
        if self._index >= self._ticks:
            return None
        shift = Decimal("0.005") * Decimal(self._index % 40)
        base = Decimal("100.00") if instrument_uid == "uid-AAA" else Decimal("200.00")
        time_ms = self._base + self._index * 300
        if instrument_uid == "uid-BBB":
            self._index += 1
        return make_book(base + shift, time_ms, uid=instrument_uid)

    async def get_last_trades(self, instrument_uid: str, date_from=None, date_to=None):
        return []


def test_dashboard_serves_live_state(tmp_path):
    """Сервис поднимает HTTP-слой и публикует стакан обеих ног."""

    async def scenario():
        leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
        leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
        config = _no_warmup(make_config(tmp_path, make_preset(leg1, leg2), _params()))
        preset = config.preset()
        port = 8129
        service = LiveWatchService(
            config, preset, _FakeWatchClient(), interval_ms=100, depth=5, port=port
        )
        stop = asyncio.Event()

        async def stopper():
            await asyncio.sleep(1.5)
            stop.set()

        asyncio.create_task(stopper())
        await service.run(duration_seconds=None, stop_event=stop)

        # Пока сервис работал, состояние публиковалось; проверяем через API после остановки
        # нельзя (сервер закрыт), поэтому читаем снимок изнутри.
        snapshot = service._state.snapshot()  # noqa: SLF001
        assert snapshot["status"] == "live"
        assert snapshot["books"]["leg1"]["bestBid"] is not None
        assert snapshot["books"]["leg2"]["bestAsk"] is not None
        assert snapshot["metrics"]["beta"] > 0
        assert snapshot["runtime"]["snapshots"] > 0
        assert "counters" in snapshot["bot"]

    asyncio.run(scenario())


def test_dashboard_http_endpoint_while_running(tmp_path):
    """Пока сервис работает, /api/state отдаёт JSON со стаканом."""

    async def scenario():
        leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
        leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
        config = _no_warmup(make_config(tmp_path, make_preset(leg1, leg2), _params()))
        preset = config.preset()
        port = 8130
        service = LiveWatchService(
            config, preset, _FakeWatchClient(), interval_ms=100, depth=5, port=port
        )
        stop = asyncio.Event()

        async def stopper():
            await asyncio.sleep(2.0)
            stop.set()

        async def fetch():
            await asyncio.sleep(0.8)
            url = f"http://127.0.0.1:{port}/api/state"
            return await asyncio.to_thread(
                lambda: json.loads(urllib.request.urlopen(url, timeout=5).read().decode("utf-8"))
            )

        fetcher = asyncio.create_task(fetch())
        stopper_task = asyncio.create_task(stopper())
        await service.run(duration_seconds=None, stop_event=stop)
        payload = await fetcher
        await stopper_task
        assert payload["status"] == "live"
        assert payload["preset"] == preset.name
        assert payload["books"]["leg1"]["bids"]
        assert "history" in payload and "prints" in payload

    asyncio.run(scenario())


def test_dispatcher_survives_client_failures(tmp_path):
    """Падение REST не завершает цикл: сервис продолжает публиковать состояние."""

    class _FlakyClient(_FakeWatchClient):
        def __init__(self):
            super().__init__(ticks=10_000)
            self.calls = 0

        async def get_order_book(self, instrument_uid: str, depth: int = 20):
            self.calls += 1
            if self.calls % 3 == 0:
                raise TimeoutError("read timed out")
            return await super().get_order_book(instrument_uid, depth)

    async def scenario():
        leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
        leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
        config = _no_warmup(make_config(tmp_path, make_preset(leg1, leg2), _params()))
        preset = config.preset()
        service = LiveWatchService(
            config, preset, _FlakyClient(), interval_ms=100, depth=5, port=8131
        )
        stop = asyncio.Event()

        async def stopper():
            await asyncio.sleep(1.6)
            stop.set()

        asyncio.create_task(stopper())
        await service.run(duration_seconds=None, stop_event=stop)
        snapshot = service._state.snapshot()  # noqa: SLF001
        assert snapshot["status"] == "live"
        assert snapshot["runtime"]["restErrors"] > 0, "ошибки должны считаться"
        assert snapshot["runtime"]["snapshots"] > 0, "сервис обязан продолжать работать"

    asyncio.run(scenario())


def _metrics(z: float, spread: float, time_ms: int):
    from types import SimpleNamespace

    return SimpleNamespace(
        zscore=z,
        spread_rub=spread,
        mean_rub=0.1,
        sigma_rub=0.12,
        time_ms=time_ms,
        obi_leg1=0.1,
    )


def test_day_tracker_collects_daily_extremes(tmp_path):
    """Трекер дня запоминает максимум |Z| и число замеров выше порога."""
    from bot_a.services.live_dashboard import _DayTracker
    from bot_a.utils.time_provider import MSK, msk_to_ms
    from datetime import datetime

    base = msk_to_ms(datetime(2026, 10, 26, 10, 0, tzinfo=MSK))
    tracker = _DayTracker("VTBR_SBER", tmp_path)
    for index, z in enumerate([0.4, -1.1, 2.2, 3.1, -0.5]):
        tracker.feed(_metrics(z, 0.2, base + index * 5_000), ready=(z == 3.1), z_entry=2.8)
    payload = tracker.payload(2.8)
    assert payload["date"] == "2026-10-26"
    assert payload["samples"] == 5
    assert payload["maxAbs"] == 3.1
    assert payload["aboveThreshold"] == 1
    assert payload["readySamples"] == 1
    assert payload["zMin"] == -1.1 and payload["zMax"] == 3.1


def test_day_tracker_persists_sample_trail(tmp_path):
    """Трейл пишется с каденцией статистики (5 с), а не раз в минуту."""
    from bot_a.services.live_dashboard import _DayTracker
    from bot_a.utils.time_provider import MSK, msk_to_ms
    from datetime import datetime

    base = msk_to_ms(datetime(2026, 10, 26, 10, 0, tzinfo=MSK))
    tracker = _DayTracker("VTBR_SBER", tmp_path)
    for step in range(4):
        tracker.feed(_metrics(0.5 + step, 0.2, base + step * 5_000), ready=False, z_entry=2.8)
    path = tmp_path / "watch_VTBR_SBER_2026-10-26.jsonl"
    assert path.exists(), "трейл дня должен сохраняться на диск"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 4, lines
    assert [round(line["z"], 1) for line in lines] == [0.5, 1.5, 2.5, 3.5]


def test_day_tracker_resets_on_new_day(tmp_path):
    """При смене торгового дня статистика начинается заново."""
    from bot_a.services.live_dashboard import _DayTracker
    from bot_a.utils.time_provider import MSK, msk_to_ms
    from datetime import datetime

    tracker = _DayTracker("VTBR_SBER", tmp_path)
    tracker.feed(_metrics(3.5, 0.2, msk_to_ms(datetime(2026, 10, 26, 23, 0, tzinfo=MSK))), False, 2.8)
    tracker.feed(_metrics(0.3, 0.2, msk_to_ms(datetime(2026, 10, 27, 10, 0, tzinfo=MSK))), False, 2.8)
    payload = tracker.payload(2.8)
    assert payload["date"] == "2026-10-27"
    assert payload["samples"] == 1
    assert payload["maxAbs"] == 0.3
