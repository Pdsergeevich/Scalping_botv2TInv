"""Тесты авто-записи живого L2 в `--preset ALL_ACTIVE` (v2.6.6).

Задача патча: каждая пара стаканов, прошедшая `PairBookSyncer`, должна попадать
в `recordings/{пресет}_{YYYY-MM-DD}.jsonl.gz` — ровно в том формате, который читает
`--mode backtest --source recordings` (`TickReplayBacktester`).

Проверяем полный цикл: минута работы `MultiPresetRunner` → файл на диске → реплей.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from dataclasses import replace
from datetime import datetime

import pytest

from bot_a.backtest.tick_replay import JsonlRecordingSource, TickReplayBacktester
from bot_a.services.multi_preset import MultiPresetRunner
from bot_a.utils.time_provider import MSK
from tests.helpers import make_book, make_config, make_meta, make_preset
from tests.test_engine import _params

START_MS = 1_700_000_000_000
POLL_INTERVAL_MS = 5000
POLLS = 12  # 12 циклов по 5 секунд книги = минута


class _ScriptedClient:
    """Отдаёт стаканы обеих ног по заранее заданному сценарию (без сети)."""

    def __init__(self, uid1: str, uid2: str, polls: int) -> None:
        self._uid1 = uid1
        self._uid2 = uid2
        self._polls = polls
        self.calls = 0

    async def get_order_book(self, uid: str, depth: int = 20):  # noqa: ARG002
        index = self.calls // 2
        if uid == self._uid1:
            price = 100.00 + index * 0.01
        elif uid == self._uid2:
            price = 101.00 + index * 0.01
        else:
            raise AssertionError(f"неизвестный uid {uid}")
        self.calls += 1
        return make_book(
            round(price, 2),
            START_MS + index * POLL_INTERVAL_MS,
            uid=uid,
        )


def _runner(tmp_path, record_l2: bool = True, name: str = ""):
    leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
    leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
    preset = make_preset(leg1, leg2)
    if name:
        preset = replace(preset, name=name)
    config = make_config(tmp_path, preset, _params())
    config = replace(
        config,
        execution=replace(config.execution, bootstrap_warmup_days=0),
    )
    runner = MultiPresetRunner(
        config,
        [preset],
        client=_ScriptedClient("uid-AAA", "uid-BBB", POLLS),
        interval_ms=100,
        serve_dashboard=False,
        record_l2=record_l2,
    )
    return runner, config, preset


def _recorded_file(config, preset) -> "object":
    directory = config.path(config.storage.recordings_dir)
    today = datetime.now(tz=MSK).strftime("%Y-%m-%d")
    return directory / f"{preset.name}_{today}.jsonl.gz"


def test_minute_of_runner_output_is_replayable(tmp_path):
    """Минута синхронизированных стаканов → файл дня → `TickReplayBacktester` без ошибок."""
    runner, config, preset = _runner(tmp_path)

    async def scenario() -> None:
        runtime = await runner._build_runtime(preset)  # noqa: SLF001
        assert runtime is not None
        assert runtime.recorder is not None, "запись должна включиться вместе с раннером"
        for _ in range(POLLS):
            await runner._poll_pair(runtime)  # noqa: SLF001
        await runner._close_recorder(runtime)  # noqa: SLF001

    asyncio.run(scenario())

    path = _recorded_file(config, preset)
    assert path.exists(), f"файл записи не создан: {path}"

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    assert len(lines) == POLLS * 2, f"ожидалось {POLLS * 2} строк (2 ноги × {POLLS})"
    assert {row["k"] for row in lines} == {"book"}
    assert {row["leg"] for row in lines} == {1, 2}
    assert all(row["b"] and row["a"] for row in lines), "стаканы должны быть непустыми"
    assert [row["t"] for row in lines] == sorted(row["t"] for row in lines)

    source = JsonlRecordingSource.from_path(path.parent)
    backtester = TickReplayBacktester(config, preset, source, params=_params())
    result = asyncio.run(backtester.run(write_report=False))

    assert result.events_processed == POLLS * 2
    assert result.totals.trades == 0  # минута спокойного рынка — сделок нет


def test_recording_disabled_writes_nothing(tmp_path):
    """`record_l2=False` не должен создавать файлы (поведение до патча)."""
    runner, config, preset = _runner(tmp_path, record_l2=False)

    async def scenario() -> None:
        runtime = await runner._build_runtime(preset)  # noqa: SLF001
        for _ in range(3):
            await runner._poll_pair(runtime)  # noqa: SLF001
        assert runtime.recorder is None
        assert runtime.recorded_books == 0

    asyncio.run(scenario())
    assert not _recorded_file(config, preset).exists()


def test_skewed_books_are_not_recorded(tmp_path):
    """Рассинхронизированные пары не пишем — иначе реплей получит фантомные выбросы Z."""
    runner, config, preset = _runner(tmp_path)

    class _SkewedClient(_ScriptedClient):
        async def get_order_book(self, uid: str, depth: int = 20):  # noqa: ARG002
            book = await super().get_order_book(uid, depth)
            if uid == "uid-BBB":
                # Вторая нога «отстала» на 2 секунды — больше лимита 1.5 интервала.
                book = make_book(book.bids[0].price, book.time_ms - 2000, uid=uid)
            return book

    runner._client = _SkewedClient("uid-AAA", "uid-BBB", POLLS)  # noqa: SLF001

    async def scenario() -> None:
        runtime = await runner._build_runtime(preset)  # noqa: SLF001
        for _ in range(5):
            await runner._poll_pair(runtime)  # noqa: SLF001
        assert runtime.syncer.skew_skips >= 5
        assert runtime.recorded_books == 0
        await runner._close_recorder(runtime)  # noqa: SLF001

    asyncio.run(scenario())


@pytest.mark.parametrize("name", ["VTBR_SBER", "AFKS_SBER"])
def test_daily_filename_matches_replay_convention(tmp_path, name):
    """Имя файла — `{пресет}_{YYYY-MM-DD}.jsonl.gz` (как в задаче по записи 3–5 дней)."""
    runner, _config, preset = _runner(tmp_path, record_l2=True, name=name)

    async def scenario() -> str:
        runtime = await runner._build_runtime(preset)  # noqa: SLF001
        path = str(runtime.recorder.file_path) if runtime.recorder else ""
        await runner._close_recorder(runtime)  # noqa: SLF001
        return path

    path = asyncio.run(scenario())
    today = datetime.now(tz=MSK).strftime("%Y-%m-%d")
    assert path.endswith(f"{name}_{today}.jsonl.gz"), path
