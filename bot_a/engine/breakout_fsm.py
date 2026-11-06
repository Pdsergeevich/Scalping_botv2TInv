"""Стратегия v3.0: пробой сжатия волатильности (momentum) на фьючерсах FORTS.

Ядро не знает, откуда пришли бары (ISS-история, реплей или живой поток) — оно
принимает завершённые минутные свечи и возвращает решения. Исполнение (сделки, комиссии,
ГО) делает вызывающая сторона: бэктестер считает по следующему бару, живой сервис — по
рыночным заявкам.

Почему не парный возврат к среднему (v2.x): комиссия T-Invest `0.025 %` за сторону
съедает `0.05 %` круга, а средняя амплитуда возврата у ликвидных акций MOEX днём —
`0.05–0.1 %`. Из 7 дней реальных L2 получилось 3 сделки. Здесь ставим на **асимметрию**:
маленький стоп (`1.2 ATR`) и неограниченный трендовый выход (Chandelier `2.0 ATR`).

Индикаторы считаются O(1) на бар:

* **Donchian(N)**: `H = max(High[t-1..t-N])`, `L = min(Low[t-1..t-N])`, `W = H - L`;
* **ATR(15)** по Уайлдеру в пунктах и в рублях (через `STEPPRICE` контракта);
* **SqueezeRatio** = `W / (ATR × √N)` — насколько канал уже «нормальной» волатильности;
* **EDA(20/90)** — EMA по закрытиям для фильтра тренда;
* **Объём(20)** — средний объём бара для требования «всплеска».
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Tuple

from bot_a.models.domain import InstrumentMetadata
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import MSK

TWO = Decimal(2)
ZERO = Decimal(0)


class BreakoutSide(str, Enum):
    LONG = "Long"
    SHORT = "Short"


class BreakoutExitReason(str, Enum):
    HARD_STOP = "HardStop"
    TRAIL_STOP = "TrailStop"
    BREAKEVEN_STOP = "BreakevenStop"
    SOFT_STOP = "SoftStop"
    PARTIAL = "Partial"
    FAILED_BREAKOUT = "FailedBreakout"
    TAKE_PROFIT = "TakeProfit"
    SESSION_CLOSE = "SessionClose"
    CLEARING = "Clearing"
    MAX_HOLD = "MaxHold"
    DATA_END = "DataEnd"


@dataclass(slots=True)
class BreakoutParams:
    """Параметры стратегии. Все «дорогие» пороги вынесены в конфиг."""

    # --- фильтры входа ---
    donchian_bars: int = 30
    atr_bars: int = 15
    squeeze_median_bars: int = 120
    max_squeeze_ratio: float = 1.60
    max_relative_squeeze: float = 0.0
    ema_fast_bars: int = 20
    ema_slow_bars: int = 90
    # --- фильтр «пилы» (v3.2) ---
    efficiency_ratio_bars: int = 20
    min_efficiency_ratio: float = 0.0
    min_ema_slope_atr: float = 0.0
    # --- мульти-режимный движок (v4.0) ---
    regime_mode: str = "breakout"  # "breakout" | "multi"
    pullback_enabled: bool = False
    pullback_min_er: float = 0.22
    pullback_lookback_bars: int = 4
    pullback_touch_atr: float = 0.30
    pullback_stop_atr: float = 0.5
    pullback_trail_atr: float = 3.2
    # v4.1: анти-churn — подтверждение импульса и волновой кулдаун после входа/стопа.
    pullback_require_thrust: bool = True
    pullback_thrust_min_atr: float = 0.25
    pullback_wave_cooldown_bars: int = 6
    pullback_wave_impulse_atr: float = 1.5
    fade_enabled: bool = False
    fade_max_er: float = 0.20
    fade_keltner_atr: float = 2.2
    fade_stop_atr: float = 0.3
    fade_min_target_fee_ratio: float = 5.0
    fade_rejection_close_ratio: float = 0.25
    fade_max_bars: int = 40
    volume_lookback: int = 20
    volume_multiple: float = 1.5
    close_in_range: float = 0.30
    min_atr_to_fee_ratio: float = 2.5
    obi_long_threshold: float = 0.20
    obi_short_threshold: float = -0.20
    breakout_buffer_steps: int = 1
    max_entry_extension_atr: float = 0.0
    # --- режим входа (v3.1) ---
    entry_mode: str = "bar_close"  # "bar_close" | "stop_level"
    # --- удержание через вечернюю сессию и ночь (v3.1) ---
    allow_overnight_hold: bool = False
    close_losers_before_clearing: bool = False
    loser_close_bars: int = 4
    max_hold_hours: float = 0.0
    clearing_close: str = "18:40,23:40"
    # --- окна входа (MSK, минуты от полуночи) ---
    session_windows: str = "10:15-13:45,14:10-18:20"
    trade_until: str = "18:35"
    # --- выходы ---
    initial_stop_atr_multiple: float = 1.2
    min_stop_steps: int = 6
    failed_breakout_minutes: int = 12
    failed_breakout_bars: int = 0
    failed_breakout_profit_atr: float = 0.6
    breakeven_trigger_atr: float = 1.2
    breakeven_fee_multiple: float = 1.5
    breakeven_lock_atr: float = 0.0
    # Tier-1 «мягкий» стоп (v3.2): при MFE >= soft_trigger стоп подтягивается к
    # soft_stop_atr (0 = безубыток, отрицательное = «чуть ниже входа»).
    soft_breakeven_trigger_atr: float = 0.0
    soft_stop_atr: float = 0.0
    # Частичная фиксация (v3.2): на MFE >= scale_out_atr закрыть долю позиции,
    # остаток ведём с breakeven-стопом на scale_out_stop_atr (сохраняет тренд).
    scale_out_atr: float = 0.0
    scale_out_fraction: float = 0.5
    scale_out_stop_atr: float = 0.0
    trail_trigger_atr: float = 1.8
    trail_atr_multiple: float = 2.0
    take_profit_atr: float = 5.0
    # --- предохранители ---
    max_consecutive_stops: int = 2
    cooldown_minutes: int = 45
    # --- издержки ---
    slippage_steps: int = 1

    def to_dict(self) -> Dict[str, object]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__slots__  # type: ignore[attr-defined]
        }

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, object]] = None) -> "BreakoutParams":
        if not data:
            return cls()
        base = cls()
        # Конфиг пишется в camelCase (`donchianBars`), поля — в snake_case.
        normalized = {_snake_case(str(key)): value for key, value in data.items()}
        values: Dict[str, object] = {}
        for name in cls.__slots__:  # type: ignore[attr-defined]
            if name not in normalized:
                continue
            current = getattr(base, name)
            raw = normalized[name]
            if isinstance(current, bool):
                values[name] = bool(raw)
            elif isinstance(current, int):
                values[name] = int(raw)
            elif isinstance(current, float):
                values[name] = float(raw)
            else:
                values[name] = str(raw)
        return replace(base, **values)  # type: ignore[arg-type]

    def overrides(self, **kwargs: object) -> "BreakoutParams":
        valid = {key: value for key, value in kwargs.items() if hasattr(self, key)}
        return replace(self, **valid) if valid else self


def _snake_case(name: str) -> str:
    """``donchianBars`` → ``donchian_bars`` (для чтения параметров из JSON)."""
    return "".join(f"_{char.lower()}" if char.isupper() else char for char in name)


def parse_windows(text: str) -> Tuple[Tuple[int, int], ...]:
    """``"10:15-13:45,14:10-18:20"`` → кортеж интервалов в минутах от полуночи MSK."""
    windows: List[Tuple[int, int]] = []
    for chunk in (text or "").split(","):
        chunk = chunk.strip()
        if not chunk or "-" not in chunk:
            continue
        left, _, right = chunk.partition("-")
        start = _minutes(left)
        end = _minutes(right)
        if start is None or end is None or end <= start:
            continue
        windows.append((start, end))
    return tuple(windows)


def _minutes(value: str) -> Optional[int]:
    parts = value.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def in_windows(time_ms: int, windows: Tuple[Tuple[int, int], ...]) -> bool:
    moment = datetime.fromtimestamp(int(time_ms) / 1000, tz=MSK)
    minute_of_day = moment.hour * 60 + moment.minute
    return any(start <= minute_of_day < end for start, end in windows)


@dataclass(slots=True)
class Bar:
    """Завершённый минутный бар."""

    time_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int

    @property
    def range(self) -> Decimal:
        return self.high - self.low

    @property
    def close_position(self) -> float:
        """Где закрытие внутри диапазона: 1.0 — на максимуме, 0.0 — на минимуме."""
        span = float(self.high - self.low)
        if span <= 0.0:
            return 0.5
        return float(self.close - self.low) / span


@dataclass(slots=True)
class IndicatorSnapshot:
    """Состояние индикаторов на закрытии бара (для решения и для отчёта)."""

    ready: bool = False
    donchian_high: float = 0.0
    donchian_low: float = 0.0
    channel_width: float = 0.0
    atr_points: float = 0.0
    atr_rub: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    volume_avg: float = 0.0
    squeeze_ratio: float = 0.0
    median_width: float = 0.0
    efficiency_ratio: float = 0.0
    ema_slope_atr: float = 0.0


class BreakoutIndicators:
    """O(1) индикаторы на завершённых барах."""

    def __init__(self, params: BreakoutParams, rub_per_price_unit: Decimal) -> None:
        self._p = params
        self._rub_per_point = dec(rub_per_price_unit) if rub_per_price_unit > 0 else Decimal(1)
        self._highs: Deque[float] = deque(maxlen=params.donchian_bars)
        self._lows: Deque[float] = deque(maxlen=params.donchian_bars)
        self._widths: Deque[float] = deque(maxlen=params.squeeze_median_bars)
        self._volumes: Deque[float] = deque(maxlen=params.volume_lookback)
        self._closes: Deque[float] = deque(maxlen=max(params.efficiency_ratio_bars + 1, 2))
        self._ema_slope_history: Deque[float] = deque(maxlen=12)
        self._atr: Optional[float] = None
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._prev_close: Optional[float] = None
        self._alpha_fast = 2.0 / (params.ema_fast_bars + 1.0)
        self._alpha_slow = 2.0 / (params.ema_slow_bars + 1.0)
        self._bars = 0
        self._snapshot = IndicatorSnapshot()

    @property
    def snapshot(self) -> IndicatorSnapshot:
        return self._snapshot

    @property
    def bars(self) -> int:
        return self._bars

    def update(self, bar: Bar) -> IndicatorSnapshot:
        high = float(bar.high)
        low = float(bar.low)
        close = float(bar.close)
        # ATR по Уайлдеру (сглаживание 1/N по TR).
        if self._prev_close is None:
            true_range = high - low
        else:
            true_range = max(
                high - low, abs(high - self._prev_close), abs(low - self._prev_close)
            )
        if self._atr is None:
            self._atr = true_range
        else:
            self._atr += (true_range - self._atr) / max(self._p.atr_bars, 1)
        self._ema_fast = (
            close if self._ema_fast is None else self._ema_fast + self._alpha_fast * (close - self._ema_fast)
        )
        self._ema_slow = (
            close if self._ema_slow is None else self._ema_slow + self._alpha_slow * (close - self._ema_slow)
        )
        self._prev_close = close
        volume_avg = (sum(self._volumes) / len(self._volumes)) if self._volumes else 0.0
        self._volumes.append(float(bar.volume))
        self._closes.append(close)
        efficiency_ratio = _efficiency_ratio(self._closes)
        self._ema_slope_history.append(float(self._ema_slow or 0.0))
        # Donchian считается по ПРЕДЫДУЩИМ барам: пробой фиксируем относительно него.
        donchian_high = max(self._highs) if len(self._highs) >= self._p.donchian_bars else 0.0
        donchian_low = min(self._lows) if len(self._lows) >= self._p.donchian_bars else 0.0
        width = (donchian_high - donchian_low) if donchian_high > 0 else 0.0
        self._highs.append(high)
        self._lows.append(low)
        if width > 0:
            self._widths.append(width)
        self._bars += 1
        ready = (
            len(self._highs) >= self._p.donchian_bars
            and self._atr is not None
            and self._atr > 0
            and len(self._widths) >= min(self._p.squeeze_median_bars, 20)
        )
        atr_points = float(self._atr or 0.0)
        median_width = _median(self._widths)
        squeeze = 0.0
        if atr_points > 0 and width > 0:
            squeeze = width / (atr_points * math.sqrt(max(self._p.donchian_bars, 1)))
        self._snapshot = IndicatorSnapshot(
            ready=ready,
            donchian_high=donchian_high,
            donchian_low=donchian_low,
            channel_width=width,
            atr_points=atr_points,
            atr_rub=atr_points * float(self._rub_per_point),
            ema_fast=float(self._ema_fast or 0.0),
            ema_slow=float(self._ema_slow or 0.0),
            volume_avg=volume_avg,
            squeeze_ratio=squeeze,
            median_width=median_width,
            efficiency_ratio=efficiency_ratio,
            ema_slope_atr=_ema_slope_atr(self._ema_slope_history, atr_points),
        )
        return self._snapshot


def _efficiency_ratio(closes: Deque[float]) -> float:
    """Kaufman ER: |C_t − C_{t−M}| / Σ|C_i − C_{i−1}| (0 = пила, 1 = чистый тренд)."""
    if len(closes) < 3:
        return 0.0
    values = list(closes)
    path = sum(abs(values[index] - values[index - 1]) for index in range(1, len(values)))
    if path <= 1e-12:
        return 0.0
    return abs(values[-1] - values[0]) / path


def _ema_slope_atr(history: Deque[float], atr_points: float, lookback: int = 10) -> float:
    """Наклон EMA_slow за ``lookback`` баров в единицах ATR (0 = флэт)."""
    if atr_points <= 1e-12 or len(history) < 2:
        return 0.0
    past = history[0] if len(history) < lookback else history[len(history) - 1 - lookback]
    return abs(history[-1] - past) / atr_points


def _median(values: Deque[float]) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    middle = len(data) // 2
    if len(data) % 2:
        return data[middle]
    return (data[middle - 1] + data[middle]) / 2.0


@dataclass(slots=True)
class BreakoutPosition:
    """Открытая позиция: направление, цена, стоп, MFE."""

    side: BreakoutSide
    quantity: int
    entry_price: Decimal
    entry_ms: int
    atr_points: float
    stop_price: Decimal
    initial_stop: Decimal
    fee_round_trip_rub: Decimal
    mfe_points: float = 0.0
    max_price: Decimal = Decimal(0)
    min_price: Decimal = Decimal(0)
    breakeven_done: bool = False
    soft_stop_done: bool = False
    scale_out_done: bool = False
    trail_active: bool = False
    entry_bar_index: int = 0
    style: str = "trend"  # "trend" | "pullback" | "fade"
    trail_mult: float = 0.0  # 0 = брать trail_atr_multiple из параметров
    target_price: Decimal = Decimal(0)  # цель для fade-сделок (EMA20)

    def unrealized_points(self, price: Decimal) -> float:
        sign = 1.0 if self.side is BreakoutSide.LONG else -1.0
        return sign * float(price - self.entry_price)

    def update_extremes(self, bar: Bar) -> None:
        if self.side is BreakoutSide.LONG:
            self.max_price = max(self.max_price, bar.high)
            self.mfe_points = max(self.mfe_points, float(self.max_price - self.entry_price))
        else:
            self.min_price = min(self.min_price, bar.low) if self.min_price > 0 else bar.low
            self.mfe_points = max(self.mfe_points, float(self.entry_price - self.min_price))


@dataclass(slots=True)
class BreakoutEvent:
    """Событие стратегии для исполнителя."""

    kind: str  # "entry" | "exit" | "partial"
    time_ms: int
    side: BreakoutSide
    price: Decimal
    reason: Optional[BreakoutExitReason] = None
    note: str = ""
    quantity: int = 0  # 0 = вся позиция; для "partial" — сколько контрактов закрыть


@dataclass(slots=True)
class BreakoutCounters:
    signals: int = 0
    entries: int = 0
    exits: int = 0
    squeeze_skips: int = 0
    trend_skips: int = 0
    volume_skips: int = 0
    wick_skips: int = 0
    obi_skips: int = 0
    fee_skips: int = 0
    window_skips: int = 0
    cooldown_skips: int = 0
    extension_skips: int = 0
    chop_skips: int = 0
    pullback_entries: int = 0
    fade_entries: int = 0
    thrust_skips: int = 0
    wave_skips: int = 0
    consecutive_stop_days: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "signals": self.signals,
            "entries": self.entries,
            "exits": self.exits,
            "squeezeSkips": self.squeeze_skips,
            "trendSkips": self.trend_skips,
            "volumeSkips": self.volume_skips,
            "wickSkips": self.wick_skips,
            "obiSkips": self.obi_skips,
            "feeSkips": self.fee_skips,
            "windowSkips": self.window_skips,
            "cooldownSkips": self.cooldown_skips,
            "extensionSkips": self.extension_skips,
            "chopSkips": self.chop_skips,
            "pullbackEntries": self.pullback_entries,
            "thrustSkips": self.thrust_skips,
            "waveSkips": self.wave_skips,
            "fadeEntries": self.fade_entries,
        }


class MomentumBreakoutStateMachine:
    """Пробойная ФСМ: сжатие → пробой с объёмом → стоп/трейл/срез.

    Не торгует сама: `on_bar` возвращает список `BreakoutEvent`, а исполнение
    (цена входа, комиссия, проскальзывание) остаётся за вызывающей стороной.
    """

    def __init__(
        self,
        meta: InstrumentMetadata,
        params: Optional[BreakoutParams] = None,
        fee_rate_per_side: float = 0.00025,
        logger: Optional[object] = None,
        fee_calculator: Optional[Callable[[Decimal, int], Decimal]] = None,
    ) -> None:
        self._meta = meta
        self._p = params or BreakoutParams()
        self._windows = parse_windows(self._p.session_windows)
        self._trade_until = _minutes(self._p.trade_until) or (18 * 60 + 35)
        self._indicators = BreakoutIndicators(self._p, meta.rub_per_price_unit)
        self._fee_rate = float(fee_rate_per_side)
        self._fee_calculator = fee_calculator
        self._log = logger
        self.position: Optional[BreakoutPosition] = None
        self.counters = BreakoutCounters()
        self._bar_index = 0
        self._cooldown_until_ms = 0
        self._consecutive_stops = 0
        self._stops_day: Optional[int] = None
        self.events: List[BreakoutEvent] = []
        # v3.1: взведённый стоп-уровень для входа «по касанию» (entryMode = stop_level)
        self._armed_side: Optional[BreakoutSide] = None
        self._armed_trigger: Decimal = ZERO
        self._armed_atr: float = 0.0
        self._armed_bar_ms: int = 0
        # v3.1: уже срезанные «предклиринговые» убытки (день + минута отсечки)
        self._loser_cuts: set = set()
        self._bar_minutes = 1
        self._last_bar_ms = 0
        # v4.0: предыдущий бар и диапазон откатов для Engine B/C.
        self._prev_bar: Optional[Bar] = None
        self._prev_snapshot: Optional[IndicatorSnapshot] = None
        # v4.1: волновой кулдаун откатов (сторона, номер бара и цена «блока»).
        self._pullback_block_side: Optional[BreakoutSide] = None
        self._pullback_block_until: int = 0
        self._pullback_block_price: float = 0.0
        self._recent_lows: Deque[float] = deque(maxlen=max(int(params.pullback_lookback_bars), 2))
        self._recent_highs: Deque[float] = deque(maxlen=max(int(params.pullback_lookback_bars), 2))
        self._clearing_minutes = tuple(
            minute
            for minute in (_minutes(chunk) for chunk in str(self._p.clearing_close).split(","))
            if minute is not None
        )

    # ------------------------------------------------------------------ оценка издержек
    def _fee_round_trip(self, price: Decimal, quantity: int) -> Decimal:
        if self._fee_calculator is not None:
            return self._fee_calculator(price, quantity)
        notional = self._meta.value_in_rub(price) * max(int(quantity), 1)
        return dec(notional * dec(self._fee_rate) * TWO)

    def _slippage_rub(self, quantity: int) -> Decimal:
        steps = Decimal(max(int(self._p.slippage_steps), 0))
        return self._meta.min_price_increment_amount * steps * Decimal(max(int(quantity), 1))

    # ------------------------------------------------------------------ основной вход
    def on_bar(self, bar: Bar, obi: Optional[float] = None) -> List[BreakoutEvent]:
        """Обрабатывает завершённый бар и возвращает решения по нему."""
        if self._last_bar_ms and bar.time_ms > self._last_bar_ms:
            self._bar_minutes = max(
                1, int(round((bar.time_ms - self._last_bar_ms) / 60_000.0))
            )
        self._last_bar_ms = bar.time_ms
        self._roll_day(bar.time_ms)
        snapshot = self._indicators.update(bar)
        self._bar_index += 1
        events: List[BreakoutEvent] = []
        if self.position is not None:
            events.extend(self._manage(bar, snapshot))
        if self.position is None:
            event = self._maybe_enter(bar, snapshot, obi)
            if event is not None:
                events.append(event)
        self.events.extend(events)
        self._prev_bar = bar
        self._prev_snapshot = snapshot
        self._recent_lows.append(float(bar.low))
        self._recent_highs.append(float(bar.high))
        return events

    def _roll_day(self, time_ms: int) -> None:
        day = datetime.fromtimestamp(int(time_ms) / 1000, tz=MSK).toordinal()
        if self._stops_day != day:
            self._stops_day = day
            self._consecutive_stops = 0
            self._cooldown_until_ms = 0

    # ------------------------------------------------------------------ вход
    def _maybe_enter(
        self, bar: Bar, snapshot: IndicatorSnapshot, obi: Optional[float]
    ) -> Optional[BreakoutEvent]:
        if str(self._p.entry_mode).lower() == "stop_level":
            return self._stop_level_entry(bar, snapshot, obi)
        return self._bar_close_entry(bar, snapshot, obi)

    def _common_setup_ok(
        self, bar: Bar, snapshot: IndicatorSnapshot
    ) -> Optional[str]:
        """Общие гейты входа. Возвращает ``None``, если вход разрешён, иначе причину."""
        params = self._p
        if not snapshot.ready:
            return "not_ready"
        self.counters.signals += 1
        if not in_windows(bar.time_ms, self._windows):
            self.counters.window_skips += 1
            return "window"
        if not params.allow_overnight_hold:
            moment = datetime.fromtimestamp(bar.time_ms / 1000, tz=MSK)
            if moment.hour * 60 + moment.minute >= self._trade_until:
                # Интрадей-режим: после времени закрытия новых входов нет.
                self.counters.window_skips += 1
                return "window"
        if bar.time_ms < self._cooldown_until_ms:
            self.counters.cooldown_skips += 1
            return "cooldown"
        if self._consecutive_stops >= max(int(params.max_consecutive_stops), 1):
            self.counters.cooldown_skips += 1
            return "cooldown"
        if params.max_squeeze_ratio > 0 and snapshot.squeeze_ratio > params.max_squeeze_ratio:
            self.counters.squeeze_skips += 1
            return "squeeze"
        if (
            params.max_relative_squeeze > 0
            and snapshot.median_width > 0
            and snapshot.channel_width > params.max_relative_squeeze * snapshot.median_width
        ):
            self.counters.squeeze_skips += 1
            return "squeeze"
        if snapshot.ema_slow <= 0:
            return "trend"
        # v3.2: фильтр «пилы». Если рынок болтается без направленного движения
        # (ER ≈ 0) и EMA90 стоит на месте, пробой канала — почти всегда ложный.
        if params.min_efficiency_ratio > 0 and snapshot.efficiency_ratio < params.min_efficiency_ratio:
            self.counters.chop_skips += 1
            return "chop"
        if params.min_ema_slope_atr > 0 and snapshot.ema_slope_atr < params.min_ema_slope_atr:
            self.counters.chop_skips += 1
            return "chop"
        fair_fee = self._fee_round_trip(bar.close, 1)
        if fair_fee <= 0 or snapshot.atr_rub / float(fair_fee) < params.min_atr_to_fee_ratio:
            self.counters.fee_skips += 1
            return "fee"
        return None

    def _open_position(
        self,
        bar: Bar,
        side: BreakoutSide,
        entry_price: Decimal,
        atr_points: float,
        note: str,
        stop_price: Optional[Decimal] = None,
        style: str = "trend",
        trail_mult: float = 0.0,
        target_price: Optional[Decimal] = None,
    ) -> BreakoutEvent:
        """Создаёт позицию и возвращает событие входа."""
        step = float(self._meta.min_price_increment)
        stop_points = max(
            self._p.initial_stop_atr_multiple * atr_points,
            step * max(int(self._p.min_stop_steps), 1),
        )
        if stop_price is None or stop_price <= 0:
            stop_price = (
                entry_price - dec(stop_points)
                if side is BreakoutSide.LONG
                else entry_price + dec(stop_points)
            )
        position = BreakoutPosition(
            side=side,
            quantity=1,
            entry_price=entry_price,
            entry_ms=bar.time_ms,
            atr_points=atr_points,
            stop_price=stop_price,
            initial_stop=stop_price,
            fee_round_trip_rub=self._fee_round_trip(entry_price, 1),
            max_price=entry_price if side is BreakoutSide.LONG else Decimal(0),
            min_price=entry_price if side is BreakoutSide.SHORT else Decimal(0),
            entry_bar_index=self._bar_index,
            style=style,
            trail_mult=trail_mult,
            target_price=target_price or Decimal(0),
        )
        self.position = position
        self.counters.entries += 1
        self._armed_side = None
        return BreakoutEvent(
            kind="entry",
            time_ms=bar.time_ms,
            side=side,
            price=entry_price,
            note=f"{note} atr={atr_points:.4f} stop={stop_price}",
        )

    def _bar_close_entry(
        self, bar: Bar, snapshot: IndicatorSnapshot, obi: Optional[float]
    ) -> Optional[BreakoutEvent]:
        """Классический вход по закрытию бара пробоя (v3.0)."""
        params = self._p
        if self._common_setup_ok(bar, snapshot) is not None:
            return None
        close = float(bar.close)
        step = float(self._meta.min_price_increment)
        buffer_points = step * max(int(params.breakout_buffer_steps), 1)
        volume_ok = float(bar.volume) >= params.volume_multiple * max(snapshot.volume_avg, 1.0)
        long_break = close >= snapshot.donchian_high + buffer_points
        short_break = close <= snapshot.donchian_low - buffer_points
        side: Optional[BreakoutSide] = None
        # Анти-«догон»: не входим, если пробойный бар уже ушёл слишком далеко от канала.
        if params.max_entry_extension_atr > 0 and snapshot.atr_points > 0:
            limit = params.max_entry_extension_atr * snapshot.atr_points
            if long_break and close - snapshot.donchian_high > limit:
                self.counters.extension_skips += 1
                return None
            if short_break and snapshot.donchian_low - close > limit:
                self.counters.extension_skips += 1
                return None
        if long_break and close > snapshot.ema_slow and snapshot.ema_fast >= snapshot.ema_slow:
            if not volume_ok:
                self.counters.volume_skips += 1
                return None
            if bar.close_position < 1.0 - params.close_in_range:
                self.counters.wick_skips += 1
                return None
            if obi is not None and obi < params.obi_long_threshold:
                self.counters.obi_skips += 1
                return None
            side = BreakoutSide.LONG
        elif short_break and close < snapshot.ema_slow and snapshot.ema_fast <= snapshot.ema_slow:
            if not volume_ok:
                self.counters.volume_skips += 1
                return None
            if bar.close_position > params.close_in_range:
                self.counters.wick_skips += 1
                return None
            if obi is not None and obi > params.obi_short_threshold:
                self.counters.obi_skips += 1
                return None
            side = BreakoutSide.SHORT
        else:
            if long_break or short_break:
                self.counters.trend_skips += 1
            return None
        return self._open_position(
            bar,
            side,
            bar.close,
            snapshot.atr_points,
            note=(
                f"squeeze={snapshot.squeeze_ratio:.2f} "
                f"vol×{float(bar.volume) / max(snapshot.volume_avg, 1.0):.2f}"
            ),
        )

    # ------------------------------------------------------------------ вход по касанию уровня
    def _stop_level_entry(
        self, bar: Bar, snapshot: IndicatorSnapshot, obi: Optional[float]
    ) -> Optional[BreakoutEvent]:
        """Вход стоп-заявкой на `H + 1 шаг` / `L - 1 шаг` по уровню предыдущего бара (v3.1).

        Логика двухфазная: на закрытии бара `t-1` уровень взводится (если выполнены
        тренд/сжатие/ATR-к-комиссии), на баре `t` проверяется касание. Цена входа —
        `max(open, trigger)` (LONG) с отсечкой гэпа больше `maxEntryExtensionAtr × ATR`.
        """
        params = self._p
        step = dec(self._meta.min_price_increment)
        # 1) Проверяем взведённый уровень на текущем баре (до обновления канала этим баром).
        if self._armed_side is not None:
            trigger = self._armed_trigger
            atr_points = self._armed_atr
            side = self._armed_side
            limit = (
                dec(params.max_entry_extension_atr * atr_points)
                if params.max_entry_extension_atr > 0 and atr_points > 0
                else dec(atr_points)
            )
            touched = bar.high >= trigger if side is BreakoutSide.LONG else bar.low <= trigger
            if not touched:
                self._armed_side = None
            else:
                gap = bar.open - trigger if side is BreakoutSide.LONG else trigger - bar.open
                if gap > limit:
                    self.counters.extension_skips += 1
                    self._armed_side = None
                else:
                    entry_price = (
                        max(bar.open, trigger) if side is BreakoutSide.LONG else min(bar.open, trigger)
                    )
                    ready = self._common_setup_ok(bar, snapshot)
                    if ready is None:
                        return self._open_position(
                            bar,
                            side,
                            entry_price,
                            atr_points,
                            note=f"squeeze={snapshot.squeeze_ratio:.2f} stoplevel={trigger}",
                        )
                    self._armed_side = None
        # 2) Взводим уровень на следующий бар по закрытию текущего.
        if not snapshot.ready:
            return None
        if self._consecutive_stops >= max(int(params.max_consecutive_stops), 1):
            return None
        if params.max_squeeze_ratio > 0 and snapshot.squeeze_ratio > params.max_squeeze_ratio:
            return None
        if (
            params.max_relative_squeeze > 0
            and snapshot.median_width > 0
            and snapshot.channel_width > params.max_relative_squeeze * snapshot.median_width
        ):
            return None
        fair_fee = self._fee_round_trip(bar.close, 1)
        if fair_fee <= 0 or snapshot.atr_rub / float(fair_fee) < params.min_atr_to_fee_ratio:
            return None
        close = float(bar.close)
        if snapshot.ema_slow <= 0:
            return None
        # Взводим уровень только после «сильного» бара: объёмный всплеск и закрытие
        # в направлении пробоя (объём бара исполнения нам неизвестен в момент заявки).
        if float(bar.volume) < params.volume_multiple * max(snapshot.volume_avg, 1.0):
            self.counters.volume_skips += 1
            self._armed_side = None
            return None
        buffer_points = dec(
            float(step) * max(int(params.breakout_buffer_steps), 1)
        )
        if close > snapshot.ema_slow and snapshot.ema_fast >= snapshot.ema_slow:
            if bar.close_position < 1.0 - params.close_in_range:
                self.counters.wick_skips += 1
                self._armed_side = None
                return None
            self._armed_side = BreakoutSide.LONG
            self._armed_trigger = dec(snapshot.donchian_high) + buffer_points
        elif close < snapshot.ema_slow and snapshot.ema_fast <= snapshot.ema_slow:
            if bar.close_position > params.close_in_range:
                self.counters.wick_skips += 1
                self._armed_side = None
                return None
            self._armed_side = BreakoutSide.SHORT
            self._armed_trigger = dec(snapshot.donchian_low) - buffer_points
        else:
            self._armed_side = None
        self._armed_atr = snapshot.atr_points
        self._armed_bar_ms = bar.time_ms
        return None

    # ------------------------------------------------------------------ ведение позиции
    def _manage(self, bar: Bar, snapshot: IndicatorSnapshot) -> List[BreakoutEvent]:
        position = self.position
        if position is None:
            return []
        params = self._p
        events: List[BreakoutEvent] = []
        atr = position.atr_points
        # 1) Chandelier-трейл: подтягиваем стоп за экстремумом **предыдущих** баров
        #    (экстремум текущего бара учитывается после проверки стопа, иначе
        #    выход по трейлу завышался бы на баровом шуме).
        if position.mfe_points >= params.trail_trigger_atr * atr:
            position.trail_active = True
        if position.trail_active:
            distance = dec((position.trail_mult or params.trail_atr_multiple) * atr)
            if position.side is BreakoutSide.LONG:
                candidate = position.max_price - distance
                if candidate > position.stop_price:
                    position.stop_price = candidate
            else:
                candidate = position.min_price + distance
                if position.stop_price <= 0 or candidate < position.stop_price:
                    position.stop_price = candidate
        # 1b) Tier-1 (v3.2): «мягкий» стоп. Летний ложный пробой успевает уйти на
        #     +1.6…2.2 ATR и разворачивается в полный −2 ATR стоп; здесь мы не даём
        #     прибыли испариться: при MFE >= soft_trigger стоп встаёт на soft_stop_atr
        #     (0 = безубыток, отрицательное значение = чуть ниже входа).
        if params.soft_breakeven_trigger_atr > 0 and atr > 0:
            if position.mfe_points >= params.soft_breakeven_trigger_atr * atr:
                offset = dec(params.soft_stop_atr * atr)
                candidate = (
                    position.entry_price + offset
                    if position.side is BreakoutSide.LONG
                    else position.entry_price - offset
                )
                if position.side is BreakoutSide.LONG and candidate > position.stop_price:
                    position.stop_price = candidate
                    position.soft_stop_done = True
                elif position.side is BreakoutSide.SHORT and (
                    position.stop_price <= 0 or candidate < position.stop_price
                ):
                    position.stop_price = candidate
                    position.soft_stop_done = True
        # 1c) Частичная фиксация (v3.2): закрываем долю позиции на MFE >= scaleOutAtr,
        #     остаток продолжаем вести трейлом (в отличие от мягкого стопа, который
        #     закрывает всё и теряет тренд).
        if (
            params.scale_out_atr > 0
            and not position.scale_out_done
            and atr > 0
            and position.mfe_points >= params.scale_out_atr * atr
        ):
            fraction = min(max(float(params.scale_out_fraction), 0.0), 1.0)
            quantity = int(round(position.quantity * fraction))
            quantity = max(quantity, 1) if position.quantity > 1 else position.quantity
            if quantity < position.quantity:
                position.scale_out_done = True
                offset = dec(params.scale_out_stop_atr * atr)
                candidate = (
                    position.entry_price + offset
                    if position.side is BreakoutSide.LONG
                    else position.entry_price - offset
                )
                if position.side is BreakoutSide.LONG and candidate > position.stop_price:
                    position.stop_price = candidate
                elif position.side is BreakoutSide.SHORT and (
                    position.stop_price <= 0 or candidate < position.stop_price
                ):
                    position.stop_price = candidate
                target = position.entry_price + dec(params.scale_out_atr * atr) * (
                    Decimal(1) if position.side is BreakoutSide.LONG else Decimal(-1)
                )
                events.append(
                    BreakoutEvent(
                        kind="partial",
                        time_ms=bar.time_ms,
                        side=position.side,
                        price=target,
                        reason=BreakoutExitReason.PARTIAL,
                        note=f"scale_out frac={fraction:.2f} mfe={position.mfe_points:.4f}",
                        quantity=quantity,
                    )
                )
                position.quantity -= quantity
        # 2) Breakeven-ratchet: после `breakevenTriggerAtr` прибыли стоп уходит
        #    минимум за комиссию, а при `breakevenLockAtr > 0` — ещё и за долю ATR.
        if (
            not position.breakeven_done
            and position.mfe_points >= params.breakeven_trigger_atr * atr
        ):
            offset = self._fee_round_trip(bar.close, position.quantity) * dec(
                params.breakeven_fee_multiple
            )
            offset /= max(dec(self._meta.rub_per_price_unit), Decimal(1)) * Decimal(
                max(int(position.quantity), 1)
            )
            if params.breakeven_lock_atr > 0:
                offset = max(offset, dec(params.breakeven_lock_atr * atr))
            candidate = (
                position.entry_price + offset
                if position.side is BreakoutSide.LONG
                else position.entry_price - offset
            )
            if position.side is BreakoutSide.LONG and candidate > position.stop_price:
                position.stop_price = candidate
                position.breakeven_done = True
            elif position.side is BreakoutSide.SHORT and (
                position.stop_price <= 0 or candidate < position.stop_price
            ):
                position.stop_price = candidate
                position.breakeven_done = True
        # 3) Стоп/трейл: проверяем по экстремуму бара (консервативно — стоп первым).
        if position.stop_price > 0:
            hit = (
                bar.low <= position.stop_price
                if position.side is BreakoutSide.LONG
                else bar.high >= position.stop_price
            )
            if hit:
                # Гэп через стоп (актуально для переноса через ночь) исполняется по open.
                if position.side is BreakoutSide.LONG and bar.open < position.stop_price:
                    fill = bar.open
                elif position.side is BreakoutSide.SHORT and bar.open > position.stop_price:
                    fill = bar.open
                else:
                    fill = position.stop_price
                reason = (
                    BreakoutExitReason.BREAKEVEN_STOP
                    if position.breakeven_done and position.mfe_points < params.trail_trigger_atr * atr
                    else (
                        BreakoutExitReason.SOFT_STOP
                        if position.soft_stop_done and not position.trail_active
                        else (
                            BreakoutExitReason.TRAIL_STOP
                            if position.trail_active
                            else BreakoutExitReason.HARD_STOP
                        )
                    )
                )
                events.append(self._close(position, fill, bar.time_ms, reason, bar))
                return events
        # 4) Тейк-профит.
        if params.take_profit_atr > 0:
            target = dec(params.take_profit_atr * atr)
            reached = (
                bar.high >= position.entry_price + target
                if position.side is BreakoutSide.LONG
                else bar.low <= position.entry_price - target
            )
            if reached:
                price = (
                    position.entry_price + target
                    if position.side is BreakoutSide.LONG
                    else position.entry_price - target
                )
                events.append(
                    self._close(position, price, bar.time_ms, BreakoutExitReason.TAKE_PROFIT, bar)
                )
                return events
        # 5) Неудавшийся пробой: 12 минут без +0.6 ATR и в минусе.
        held_minutes = (bar.time_ms - position.entry_ms) / 60_000.0
        unrealized = position.unrealized_points(bar.close)
        fail_after = float(params.failed_breakout_minutes)
        if params.failed_breakout_bars > 0:
            # На старших таймфреймах порог задаётся в барах: 12 минут на 60-мин баре
            # означали бы выход на первом же баре после входа.
            fail_after = max(fail_after, params.failed_breakout_bars * self._bar_minutes)
        if (
            held_minutes >= fail_after
            and position.mfe_points < params.failed_breakout_profit_atr * atr
            and unrealized < 0
        ):
            events.append(
                self._close(position, bar.close, bar.time_ms, BreakoutExitReason.FAILED_BREAKOUT, bar)
            )
            return events
        # 6) Срез «проигрышных» позиций перед клирингом и лимит времени удержания (v3.1).
        moment = datetime.fromtimestamp(bar.time_ms / 1000, tz=MSK)
        minute_of_day = moment.hour * 60 + moment.minute
        bars_held = max(
            int(round((bar.time_ms - position.entry_ms) / max(self._bar_minutes * 60_000, 1))), 0
        )
        if params.max_hold_hours > 0 and held_minutes >= params.max_hold_hours * 60.0:
            events.append(
                self._close(position, bar.close, bar.time_ms, BreakoutExitReason.MAX_HOLD, bar)
            )
            return events
        if params.allow_overnight_hold and params.close_losers_before_clearing:
            for cutoff in self._clearing_minutes:
                key = (moment.date(), cutoff)
                if cutoff <= minute_of_day < cutoff + max(self._bar_minutes, 1) and (
                    key not in self._loser_cuts
                ):
                    self._loser_cuts.add(key)
                    if unrealized < 0 and bars_held >= max(int(params.loser_close_bars), 1):
                        events.append(
                            self._close(
                                position, bar.close, bar.time_ms, BreakoutExitReason.CLEARING, bar
                            )
                        )
                        return events
        # 7) Закрытие сессии — только в интрадей-режиме (без переноса через ночь).
        if not params.allow_overnight_hold and minute_of_day >= self._trade_until:
            events.append(
                self._close(position, bar.close, bar.time_ms, BreakoutExitReason.SESSION_CLOSE, bar)
            )
            return events
        # 7) Только теперь учитываем экстремум текущего бара (для следующего шага).
        position.update_extremes(bar)
        return events

    def _close(
        self,
        position: BreakoutPosition,
        price: Decimal,
        time_ms: int,
        reason: BreakoutExitReason,
        bar: Bar,
    ) -> BreakoutEvent:
        self.position = None
        self.counters.exits += 1
        if reason in (
            BreakoutExitReason.HARD_STOP,
            BreakoutExitReason.TRAIL_STOP,
            BreakoutExitReason.BREAKEVEN_STOP,
            BreakoutExitReason.SOFT_STOP,
        ):
            self._consecutive_stops += 1
            self._cooldown_until_ms = time_ms + int(self._p.cooldown_minutes) * 60_000
        else:
            self._consecutive_stops = 0
        return BreakoutEvent(
            kind="exit",
            time_ms=time_ms,
            side=position.side,
            price=price,
            reason=reason,
            note=f"mfe={position.mfe_points:.4f} atr={position.atr_points:.4f}",
        )

    def force_close(self, bar: Bar, reason: BreakoutExitReason = BreakoutExitReason.DATA_END):
        """Принудительное закрытие (конец данных/остановка)."""
        if self.position is None:
            return None
        return self._close(self.position, bar.close, bar.time_ms, reason, bar)