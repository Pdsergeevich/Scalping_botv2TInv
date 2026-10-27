"""Тесты кэша свечей MOEX ISS: текущий день обязан дочитываться по мере торгов.

Баг, который эти тесты закрывают: файл кэша за сегодня создавался один раз
(например, в 11:58) и больше не обновлялся — из-за этого `--mode paper --source iss`
и диагностика «за сегодня» видели только утро, а вечерние движения цены игнорировали.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone

from bot_a.backtest.iss_source import IssCandleSource
from bot_a.utils.time_provider import MSK, msk_to_ms
from tests.helpers import make_meta


def _row(ms: int, close: str = "100.00000") -> dict:
    return {
        "time": int(ms),
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 10,
    }


def _meta():
    return make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")


def _source_with_fake_fetch(tmp_path, rows_by_call):
    """Источник, у которого сетевой фетч подменён заранее заготовленными ответами."""
    source = IssCandleSource(tmp_path, today_ttl_seconds=120.0)
    calls: list[int] = []

    def fake_fetch_day(meta, day):
        calls.append(1)
        return rows_by_call(len(calls) - 1)

    source._fetch_day = fake_fetch_day  # noqa: SLF001 - подмена сетевого слоя
    return source, calls


def _day_ms(day: datetime, hour: int, minute: int) -> int:
    return msk_to_ms(datetime(day.year, day.month, day.day, hour, minute, tzinfo=MSK))


def test_today_cache_is_refetched_when_stale(tmp_path):
    """Кэш текущего дня старше TTL перечитывается, а не отдаётся как есть."""
    today = datetime.now(tz=MSK).date()
    t0 = _day_ms(today, 10, 0)
    t1 = _day_ms(today, 15, 0)
    source, calls = _source_with_fake_fetch(
        tmp_path,
        lambda index: [_row(t0, "100.0")] if index == 0 else [_row(t0, "100.0"), _row(t1, "101.5")],
    )
    meta = _meta()
    path = source._cache_path(meta, today)  # noqa: SLF001

    first = asyncio.run(source.get_candles(meta, msk_to_ms(datetime(today.year, today.month, today.day, tzinfo=MSK)), t0 + 3_600_000))
    assert len(first) == 1 and str(first[0].close).startswith("100.0"), "первый фетч — только утро"

    # Состариваем кэш: как будто файл записан два часа назад.
    old = time.time() - 7200
    os.utime(path, (old, old))

    second = asyncio.run(
        source.get_candles(meta, msk_to_ms(datetime(today.year, today.month, today.day, tzinfo=MSK)), t1 + 3_600_000)
    )
    assert len(calls) == 2, "устаревший кэш текущего дня должен обновляться"
    assert len(second) == 2, "вечерняя свеча обязана появиться"
    assert str(second[-1].close).startswith("101.5")


def test_today_cache_is_reused_when_fresh(tmp_path):
    """Свежий кэш текущего дня не дёргает сеть повторно (лимиты ISS)."""
    today = datetime.now(tz=MSK).date()
    t0 = _day_ms(today, 10, 0)
    source, calls = _source_with_fake_fetch(tmp_path, lambda index: [_row(t0, "100.0")])
    meta = _meta()
    start = msk_to_ms(datetime(today.year, today.month, today.day, tzinfo=MSK))
    asyncio.run(source.get_candles(meta, start, t0 + 3_600_000))
    asyncio.run(source.get_candles(meta, start, t0 + 3_600_000))
    assert len(calls) == 1, "свежий кэш должен переиспользоваться"


def test_past_day_cache_is_never_refetched(tmp_path):
    """Прошедшие дни неизменяемы: их кэш читается всегда, даже очень старый."""
    past = datetime.now(tz=MSK).date() - timedelta(days=10)
    t0 = _day_ms(past, 10, 0)
    source, calls = _source_with_fake_fetch(tmp_path, lambda index: [_row(t0, "100.0")])
    meta = _meta()
    path = source._cache_path(meta, past)  # noqa: SLF001
    path.write_text(json.dumps([_row(t0, "99.0")]), encoding="utf-8")
    old = time.time() - 30 * 86400
    os.utime(path, (old, old))
    start = msk_to_ms(datetime(past.year, past.month, past.day, tzinfo=MSK))
    candles = asyncio.run(source.get_candles(meta, start, t0 + 3_600_000))
    assert len(calls) == 0, "исторические дни из сети не перезапрашиваются"
    assert str(candles[0].close).startswith("99.0")


def test_stale_today_cache_merges_new_rows(tmp_path):
    """Обновление текущего дня дописывает новые свечи, не теряя уже скачанные."""
    today = datetime.now(tz=MSK).date()
    t0 = _day_ms(today, 10, 0)
    t1 = _day_ms(today, 16, 0)
    source, _ = _source_with_fake_fetch(
        tmp_path,
        lambda index: [_row(t0, "100.0")] if index == 0 else [_row(t1, "102.0")],
    )
    meta = _meta()
    path = source._cache_path(meta, today)  # noqa: SLF001
    start = msk_to_ms(datetime(today.year, today.month, today.day, tzinfo=MSK))
    asyncio.run(source.get_candles(meta, start, t0 + 3_600_000))
    old = time.time() - 7200
    os.utime(path, (old, old))
    candles = asyncio.run(source.get_candles(meta, start, t1 + 3_600_000))
    assert [str(item.close)[:5] for item in candles] == ["100.0", "102.0"]
