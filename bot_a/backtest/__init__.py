"""Пакет бэктеста: исторические свечи, реплей L2, запись и оптимизация."""

from bot_a.backtest.candles import Candle, InMemoryCandleSource, TInvestCandleSource
from bot_a.backtest.grid_search import GridSearchOptimizer, GridSearchResult
from bot_a.backtest.historical_candles import BacktestResult, HistoricalCandleBacktester
from bot_a.backtest.iss_source import IssCandleSource, fetch_futures_metadata, fetch_stock_metadata
from bot_a.backtest.recorder import MarketDataRecorder, RecordingService
from bot_a.backtest.simulation import SimulationOutcome, simulate_events
from bot_a.backtest.tick_replay import (
    JsonlRecordingSource,
    PostgresRecordingSource,
    ReplayRange,
    TickReplayBacktester,
)

__all__ = [
    "BacktestResult",
    "Candle",
    "GridSearchOptimizer",
    "GridSearchResult",
    "HistoricalCandleBacktester",
    "InMemoryCandleSource",
    "IssCandleSource",
    "JsonlRecordingSource",
    "MarketDataRecorder",
    "PostgresRecordingSource",
    "RecordingService",
    "ReplayRange",
    "SimulationOutcome",
    "TInvestCandleSource",
    "TickReplayBacktester",
    "fetch_futures_metadata",
    "fetch_stock_metadata",
    "simulate_events",
]
