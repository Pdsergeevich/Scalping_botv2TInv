"""Интеграционные проверки бэктеста, записи/реплея и оптимизатора."""

import asyncio
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from bot_a.backtest.grid_search import GridSearchOptimizer
from bot_a.backtest.historical_candles import HistoricalCandleBacktester
from bot_a.backtest.recorder import MarketDataRecorder
from bot_a.backtest.tick_replay import JsonlRecordingSource, ReplayRange, TickReplayBacktester
from bot_a.models.domain import ExecutionTopology
from tests.helpers import (
    make_candles,
    make_config,
    make_meta,
    make_preset,
    trading_start_ms,
)


def _pair():
    leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
    leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
    return make_preset(leg1, leg2)


def _shocked_candles(minutes: int = 120):
    start = trading_start_ms("2025-03-11", 10, 0)
    leg1 = make_candles(start, minutes, "100", "0", 2000, "0.02")
    leg2 = make_candles(start, minutes, "101", "0", 2000, "0.02")
    shifted = []
    for index, candle in enumerate(leg2):
        if 40 <= index <= 55:
            delta = Decimal("2.5")
            shifted.append(
                replace(
                    candle,
                    open=candle.open + delta,
                    high=candle.high + delta,
                    low=candle.low + delta,
                    close=candle.close + delta,
                )
            )
        else:
            shifted.append(candle)
    return start, leg1, shifted


def _events_source(preset, leg1_candles, leg2_candles):
    from bot_a.backtest.candles import InMemoryCandleSource

    return InMemoryCandleSource(
        {
            preset.leg1.ticker: leg1_candles,
            preset.leg2.ticker: leg2_candles,
        }
    )


def test_historical_backtest_produces_reports(tmp_path):
    preset = _pair()
    config = make_config(tmp_path, preset)
    start, leg1, leg2 = _shocked_candles()
    source = _events_source(preset, leg1, leg2)
    backtester = HistoricalCandleBacktester(config, preset, source)
    result = asyncio.run(
        backtester.run(
            days=1,
            start_ms=start,
            end_ms=start + (len(leg1) - 1) * 60_000,
            write_report=True,
        )
    )
    assert result.candles_leg1 == len(leg1)
    assert result.candles_leg2 == len(leg2)
    assert result.events_processed == len(leg1) * 12 * 4
    assert result.totals.total_days >= 1
    assert result.summaries[0].date == "2025-03-11"
    assert len(result.report_paths) >= 3
    for path in result.report_paths:
        assert Path(path).exists()
    assert result.totals.trades >= 1
    assert result.totals.fees_rub > 0


def test_record_and_replay_roundtrip(tmp_path):
    preset = _pair()
    config = make_config(tmp_path, preset)
    start, leg1, leg2 = _shocked_candles()
    from bot_a.backtest.historical_candles import iter_candle_events

    async def record():
        recorder = MarketDataRecorder(config, preset)
        await recorder.start()
        for event in iter_candle_events(preset.leg1, preset.leg2, leg1[:40], leg2[:40]):
            recorder.write(event)
        await recorder.close()
        return recorder.file_path, recorder.stats

    path, stats = asyncio.run(record())
    assert path is not None and Path(path).exists()
    assert stats.books == 40 * 12 * 2
    assert stats.trades == 40 * 12 * 2

    source = JsonlRecordingSource.from_path(path)
    backtester = TickReplayBacktester(config, preset, source)
    result = asyncio.run(
        backtester.run(
            ReplayRange(start, start + 40 * 60_000),
            write_report=True,
        )
    )
    assert result.events_processed == 40 * 12 * 4
    assert len(result.report_paths) >= 2


def test_grid_search_ranks_and_exports_winner(tmp_path):
    preset = _pair()
    config = make_config(tmp_path, preset)
    start, leg1, leg2 = _shocked_candles(80)
    source = _events_source(preset, leg1, leg2)
    optimizer = GridSearchOptimizer(config, preset, source, jobs=1)

    async def run():
        return await optimizer.run(
            start_ms=start,
            end_ms=start + (len(leg1) - 1) * 60_000,
            z_entries=(1.0, 1.4),
            z_exits=(0.2,),
            ewma_windows=(20,),
            topologies=(ExecutionTopology.SINGLE_LEG_LEAD_LAG, ExecutionTopology.DUAL_LEG_HEDGED),
            write_reports=True,
        )

    results = asyncio.run(run())
    assert len(results) == 4
    scores = [item.score for item in results]
    assert scores == sorted(scores, reverse=True)
    reports = list((tmp_path / "reports").glob("optimize_winner_*.json"))
    assert reports, "должен быть экспортирован пресет-победитель"
    assert all(item.totals.total_days >= 0 for item in results)
