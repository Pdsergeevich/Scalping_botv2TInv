"""Холодный старт: прогрев статистики ядра историческими свечами MOEX ISS.

Живой стрим L2 стартует в 10:00 MSK с пустыми окнами EWMA. Без прогрева бот
пропускает всё утреннее окно ``PrimeHoursOnly`` (10:05–12:30), потому что
``ewmaWindow = 2400`` в терминах 5-секундных сэмплов — это 200 минут.

``bootstrap_warmup`` проигрывает последние торговые дни через тот же
``PairSpreadCalculator``, что и live-поток, с торговлей выключенной, и оставляет
ядро в состоянии ``Idle`` с наполненными окнами ``slow``/``macro`` беты.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Sequence

from bot_a.backtest.candles import Candle
from bot_a.backtest.historical_candles import iter_candle_events
from bot_a.backtest.iss_source import IssCandleSource
from bot_a.config import AppConfig, PairPreset
from bot_a.utils.time_provider import MSK, now_ms

LOGGER_NAME = "bot_a.bootstrap"
DEFAULT_BOOTSTRAP_DAYS = 5


@dataclass(frozen=True, slots=True)
class WarmupReport:
    """Итог прогрева: состояние ядра и объём проигранной истории."""

    preset: str
    days: int
    candles_leg1: int
    candles_leg2: int
    first_candle_msk: Optional[str]
    last_candle_msk: Optional[str]
    tick_count: int
    is_warm: bool
    state: str
    mu: float
    sigma_leg1_rub: float
    beta_slow: float
    beta_macro: float
    beta_drift_pct: Optional[float]
    regime_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset,
            "days": self.days,
            "candlesLeg1": self.candles_leg1,
            "candlesLeg2": self.candles_leg2,
            "first": self.first_candle_msk,
            "last": self.last_candle_msk,
            "tickCount": self.tick_count,
            "isWarm": self.is_warm,
            "state": self.state,
            "mu": self.mu,
            "sigmaLeg1Rub": self.sigma_leg1_rub,
            "betaSlow": self.beta_slow,
            "betaMacro": self.beta_macro,
            "betaDriftPct": self.beta_drift_pct,
            "regimeSamples": self.regime_samples,
        }

    def render(self) -> str:
        drift = "н/д" if self.beta_drift_pct is None else f"{self.beta_drift_pct:+.3f}%"
        return (
            f"Warmup complete: tick_count={self.tick_count}, mu={self.mu:.4f}, "
            f"sigma_leg1={self.sigma_leg1_rub:.4f}, beta_slow={self.beta_slow:.6f}, "
            f"beta_macro={self.beta_macro:.6f}, beta_drift={drift} "
            f"(state={self.state}, свечей {self.candles_leg1}/{self.candles_leg2}, "
            f"{self.first_candle_msk}..{self.last_candle_msk})"
        )


def _trim_last_days(candles: Sequence[Candle], days: int) -> list[Candle]:
    """Оставляет свечи последних ``days`` торговых дат (по MSK)."""
    if not candles:
        return []
    dates = sorted({datetime.fromtimestamp(candle.time_ms / 1000, tz=MSK).date() for candle in candles})
    keep = set(dates[-max(int(days), 1):])
    return [
        candle
        for candle in candles
        if datetime.fromtimestamp(candle.time_ms / 1000, tz=MSK).date() in keep
    ]


def _history_window(days: int, end_ms: Optional[int] = None) -> tuple[int, int]:
    """Календарное окно под ``days`` торговых дней (с запасом на выходные)."""
    finish = int(end_ms if end_ms is not None else now_ms())
    span = timedelta(days=max(int(days), 1) * 2 + 10)
    start = datetime.fromtimestamp(finish / 1000, tz=MSK) - span
    return int(start.timestamp() * 1000), finish


async def bootstrap_warmup(
    config: AppConfig,
    preset: PairPreset,
    engine: Any,
    source: Optional[Any] = None,
    days: int = DEFAULT_BOOTSTRAP_DAYS,
    end_ms: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    clock: Optional[Any] = None,
) -> WarmupReport:
    """Проигрывает историю через готовое ядро, не открывая сделок.

    Используется в ``--mode paper`` и ``--mode live`` до подписки на стрим.
    """
    log = logger or logging.getLogger(LOGGER_NAME)
    days = max(int(days), 1)
    candles_source = source or IssCandleSource(config.path(config.storage.data_cache_dir))
    start_ms, finish_ms = _history_window(days, end_ms)

    candles1, candles2 = await asyncio.gather(
        candles_source.get_candles(preset.leg1, start_ms, finish_ms, "1min"),
        candles_source.get_candles(preset.leg2, start_ms, finish_ms, "1min"),
    )
    candles1 = _trim_last_days(candles1, days)
    candles2 = _trim_last_days(candles2, days)
    if not candles1 or not candles2:
        log.warning(
            "Прогрев пропущен: нет свечей MOEX ISS за последние %d торговых дней (кэш %s)",
            days,
            config.path(config.storage.data_cache_dir),
        )
        engine.set_bootstrap(False)
        state = engine.spread_state()
        return WarmupReport(
            preset=preset.name,
            days=0,
            candles_leg1=0,
            candles_leg2=0,
            first_candle_msk=None,
            last_candle_msk=None,
            tick_count=int(state["tickCount"]),
            is_warm=bool(state["isWarm"]),
            state=str(state["state"]),
            mu=float(state["mu"]),
            sigma_leg1_rub=float(state["sigmaLeg1Rub"]),
            beta_slow=float(state["betaSlow"]),
            beta_macro=float(state["betaMacro"]),
            beta_drift_pct=state["betaDriftPct"],
            regime_samples=int(state["regimeSamples"]),
        )

    log.info(
        "Прогрев из MOEX ISS: %s/%s, свичей %d/%d, окно %s..%s",
        preset.leg1.ticker,
        preset.leg2.ticker,
        len(candles1),
        len(candles2),
        datetime.fromtimestamp(candles1[0].time_ms / 1000, tz=MSK).strftime("%Y-%m-%d %H:%M"),
        datetime.fromtimestamp(candles1[-1].time_ms / 1000, tz=MSK).strftime("%Y-%m-%d %H:%M"),
    )

    events = iter_candle_events(
        preset.leg1,
        preset.leg2,
        candles1,
        candles2,
        depth=config.execution.order_book_depth,
        phase_lock=True,
        synth_mode="ticks",
    )
    engine.set_bootstrap(True)
    processed = 0
    try:
        for event in events:
            if clock is not None and hasattr(clock, "set"):
                clock.set(event.time_ms)
            if event.is_book and event.book is not None:
                await engine.on_order_book(event.leg, event.book)
            elif event.trade is not None:
                await engine.on_trade(event.leg, event.trade)
            processed += 1
            if processed % 5000 == 0:
                await asyncio.sleep(0)
    finally:
        engine.set_bootstrap(False)
    await engine.on_clock()

    state = engine.spread_state()
    report = WarmupReport(
        preset=preset.name,
        days=days,
        candles_leg1=len(candles1),
        candles_leg2=len(candles2),
        first_candle_msk=datetime.fromtimestamp(candles1[0].time_ms / 1000, tz=MSK).isoformat(),
        last_candle_msk=datetime.fromtimestamp(candles1[-1].time_ms / 1000, tz=MSK).isoformat(),
        tick_count=int(state["tickCount"]),
        is_warm=bool(state["isWarm"]),
        state=str(state["state"]),
        mu=float(state["mu"]),
        sigma_leg1_rub=float(state["sigmaLeg1Rub"]),
        beta_slow=float(state["betaSlow"]),
        beta_macro=float(state["betaMacro"]),
        beta_drift_pct=state["betaDriftPct"],
        regime_samples=int(state["regimeSamples"]),
    )
    log.info(report.render())
    return report
