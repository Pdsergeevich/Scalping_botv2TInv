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


class BreakoutSide(str, Enum):
    LONG = "Long"
    SHORT = "Short"


class BreakoutExitReason(str, Enum):
    HARD_STOP = "HardStop"
    TRAIL_STOP = "TrailStop"
    BREAKEVEN_STOP = "BreakevenStop"
    FAILED_BREAKOUT = "FailedBreakout"
    TAKE_PROFIT = "TakeProfit"
    SESSION_CLOSE = "SessionClose"
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
    volume_lookback: int = 20
    volume_multiple: float = 1.5
    close_in_range: float = 0.30
    min_atr_to_fee_ratio: float = 2.5
    obi_long_threshold: float = 0.20
    obi_short_threshold: float = -0.20
    breakout_buffer_steps: int = 1
    max_entry_extension_atr: float = 0.0
    # --- окна входа (MSK, минуты от полуночи) ---
    session_windows: str = "10:15-13:45,14:10-18:20"
    trade_until: str = "18:35"
    # --- выходы ---
    initial_stop_atr_multiple: float = 1.2
    min_stop_steps: int = 6
    failed_breakout_minutes: int = 12
    failed_breakout_profit_atr: float = 0.6
    breakeven_trigger_atr: float = 1.2
    breakeven_fee_multiple: float = 1.5
    breakeven_lock_atr: float = 0.0
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


class BreakoutIndicators:
    """O(1) индикаторы на завершённых барах."""

    def __init__(self, params: BreakoutParams, rub_per_price_unit: Decimal) -> None:
        self._p = params
        self._rub_per_point = dec(rub_per_price_unit) if rub_per_price_unit > 0 else Decimal(1)
        self._highs: Deque[float] = deque(maxlen=params.donchian_bars)
        self._lows: Deque[float] = deque(maxlen=params.donchian_bars)
        self._widths: Deque[float] = deque(maxlen=params.squeeze_median_bars)
        self._volumes: Deque[float] = deque(maxlen=params.volume_lookback)
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
        )
        return self._snapshot


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
    trail_active: bool = False
    entry_bar_index: int = 0

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

    kind: str  # "entry" | "exit"
    time_ms: int
    side: BreakoutSide
    price: Decimal
    reason: Optional[BreakoutExitReason] = None
    note: str = ""


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
        params = self._p
        if not snapshot.ready:
            return None
        self.counters.signals += 1
        if not in_windows(bar.time_ms, self._windows):
            self.counters.window_skips += 1
            return None
        if bar.time_ms < self._cooldown_until_ms:
            self.counters.cooldown_skips += 1
            return None
        if self._consecutive_stops >= max(int(params.max_consecutive_stops), 1):
            self.counters.cooldown_skips += 1
            return None
        if params.max_squeeze_ratio > 0 and snapshot.squeeze_ratio > params.max_squeeze_ratio:
            self.counters.squeeze_skips += 1
            return None
        # Второй, относительный фильтр: канал уже своей 120-минутной медианы.
        if (
            params.max_relative_squeeze > 0
            and snapshot.median_width > 0
            and snapshot.channel_width > params.max_relative_squeeze * snapshot.median_width
        ):
            self.counters.squeeze_skips += 1
            return None
        close = float(bar.close)
        if snapshot.ema_slow <= 0:
            return None
        fair_fee = self._fee_round_trip(bar.close, 1)
        if fair_fee <= 0 or snapshot.atr_rub / float(fair_fee) < params.min_atr_to_fee_ratio:
            self.counters.fee_skips += 1
            return None
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
        quantity = 1
        stop_points = max(
            params.initial_stop_atr_multiple * snapshot.atr_points,
            step * max(int(params.min_stop_steps), 1),
        )
        stop_price = (
            bar.close - dec(stop_points)
            if side is BreakoutSide.LONG
            else bar.close + dec(stop_points)
        )
        fee_rt = self._fee_round_trip(bar.close, quantity)
        position = BreakoutPosition(
            side=side,
            quantity=quantity,
            entry_price=bar.close,
            entry_ms=bar.time_ms,
            atr_points=snapshot.atr_points,
            stop_price=stop_price,
            initial_stop=stop_price,
            fee_round_trip_rub=fee_rt,
            max_price=bar.high if side is BreakoutSide.LONG else Decimal(0),
            min_price=bar.low if side is BreakoutSide.SHORT else Decimal(0),
            entry_bar_index=self._bar_index,
        )
        self.position = position
        self.counters.entries += 1
        return BreakoutEvent(
            kind="entry",
            time_ms=bar.time_ms,
            side=side,
            price=bar.close,
            note=(
                f"squeeze={snapshot.squeeze_ratio:.2f} atr={snapshot.atr_points:.3f} "
                f"vol×{float(bar.volume) / max(snapshot.volume_avg, 1.0):.2f} "
                f"stop={stop_price}"
            ),
        )

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
            distance = dec(params.trail_atr_multiple * atr)
            if position.side is BreakoutSide.LONG:
                candidate = position.max_price - distance
                if candidate > position.stop_price:
                    position.stop_price = candidate
            else:
                candidate = position.min_price + distance
                if position.stop_price <= 0 or candidate < position.stop_price:
                    position.stop_price = candidate
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
                reason = (
                    BreakoutExitReason.BREAKEVEN_STOP
                    if position.breakeven_done and position.mfe_points < params.trail_trigger_atr * atr
                    else (
                        BreakoutExitReason.TRAIL_STOP
                        if position.trail_active
                        else BreakoutExitReason.HARD_STOP
                    )
                )
                events.append(
                    self._close(position, position.stop_price, bar.time_ms, reason, bar)
                )
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
        if (
            held_minutes >= params.failed_breakout_minutes
            and position.mfe_points < params.failed_breakout_profit_atr * atr
            and unrealized < 0
        ):
            events.append(
                self._close(position, bar.close, bar.time_ms, BreakoutExitReason.FAILED_BREAKOUT, bar)
            )
            return events
        # 6) Закрытие сессии.
        moment = datetime.fromtimestamp(bar.time_ms / 1000, tz=MSK)
        if moment.hour * 60 + moment.minute >= self._trade_until:
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
