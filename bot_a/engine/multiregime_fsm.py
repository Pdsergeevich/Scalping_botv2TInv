"""Мульти-режимный движок v4.0: пробой + откат в тренде + фейд крайностей.

Три движка живут в одном конечном автомате и выбираются по **режиму**, который
измеряется тем же Kaufman ER, что и в v3.2:

* ``Engine A`` — пробой сжатия (наследуется от ``MomentumBreakoutStateMachine``):
  работает, когда `ER >= minEfficiencyRatio`;
* ``Engine B`` — откат к EMA20 внутри уже установившегося тренда
  (`ER >= pullbackMinEr`): даёт 2–4 входа на каждую трендовую волну вместо одного;
* ``Engine C`` — фейд растяжения за полосы Кельтнера (`ER <= fadeMaxEr`): покупаем
  возврат к EMA20 на «пиле», где пробойная логика исторически теряет.

Приоритет при выборе: C (если режим пилы) → B (если тренд и есть откат) → A.
Все три используют общий сайзинг, общий стоп-менеджер (breakeven/trail/кулдаун)
и общие правила исполнения, поэтому сравнимы на одних данных.
"""

from __future__ import annotations

from decimal import Decimal
from typing import List, Optional

from bot_a.engine.breakout_fsm import (
    Bar,
    BreakoutEvent,
    BreakoutExitReason,
    BreakoutSide,
    BreakoutParams,
    IndicatorSnapshot,
    MomentumBreakoutStateMachine,
    in_windows,
)
from bot_a.utils.quotation import dec

ZERO = Decimal(0)


class MultiRegimeStateMachine(MomentumBreakoutStateMachine):
    """FSM с тремя движками (пробой / откат / фейд)."""

    # ------------------------------------------------------------------ маршрутизация
    def _maybe_enter(
        self, bar: Bar, snapshot: IndicatorSnapshot, obi: Optional[float]
    ) -> Optional[BreakoutEvent]:
        params = self._p
        if str(params.regime_mode).lower() != "multi":
            return super()._maybe_enter(bar, snapshot, obi)
        if not snapshot.ready:
            return super()._maybe_enter(bar, snapshot, obi)
        efficiency = snapshot.efficiency_ratio
        if params.fade_enabled and efficiency <= params.fade_max_er:
            event = self._fade_entry(bar, snapshot, obi)
            if event is not None:
                return event
        if params.pullback_enabled and efficiency >= params.pullback_min_er:
            event = self._pullback_entry(bar, snapshot, obi)
            if event is not None:
                return event
        return super()._maybe_enter(bar, snapshot, obi)

    # ------------------------------------------------------------------ общие гейты
    def _entry_allowed(self, bar: Bar, snapshot: IndicatorSnapshot, obi: Optional[float]) -> bool:
        """Гейты, общие для движков B и C (Engine A проверяет их сам)."""
        params = self._p
        if not in_windows(bar.time_ms, self._windows):
            self.counters.window_skips += 1
            return False
        if self._cooldown_until_ms and bar.time_ms < self._cooldown_until_ms:
            self.counters.cooldown_skips += 1
            return False
        if self._consecutive_stops >= max(int(params.max_consecutive_stops), 1):
            return False
        if snapshot.atr_points <= 0 or snapshot.ema_fast <= 0 or snapshot.ema_slow <= 0:
            return False
        fee = self._fee_round_trip(bar.close, 1)
        if fee <= 0 or snapshot.atr_rub / float(fee) < params.min_atr_to_fee_ratio:
            self.counters.fee_skips += 1
            return False
        return True

    # ------------------------------------------------------------------ Engine B
    def _pullback_entry(
        self, bar: Bar, snapshot: IndicatorSnapshot, obi: Optional[float]
    ) -> Optional[BreakoutEvent]:
        """Откат к EMA20 внутри тренда: покупаем касание средней, а не догоняем импульс.

        Вход по закрытию бара, который **дотянулся** до EMA20 (low ≤ EMA20 + touch·ATR),
        но не сломал тренд (close выше EMA20). Стоп — за экстремум отката минус
        `pullback_stop_atr · ATR`; трейл — Chandelier `pullback_trail_atr`.
        """
        params = self._p
        if not self._entry_allowed(bar, snapshot, obi):
            return None
        atr = snapshot.atr_points
        close = float(bar.close)
        ema_fast = snapshot.ema_fast
        ema_slow = snapshot.ema_slow
        touch = params.pullback_touch_atr * atr
        trend_up = close > ema_slow and ema_fast >= ema_slow
        trend_down = close < ema_slow and ema_fast <= ema_slow
        previous = self._prev_bar
        if trend_up and float(bar.low) <= ema_fast + touch:
            if previous is not None and not self._pullback_thrust_ok(bar, atr, BreakoutSide.LONG):
                return None
            if not self._pullback_wave_ok(bar, atr, BreakoutSide.LONG):
                return None
            swing_low = min(self._recent_lows) if self._recent_lows else float(bar.low)
            stop = dec(min(swing_low, float(bar.low)) - params.pullback_stop_atr * atr)
            if stop >= bar.close:
                stop = dec(close - max(params.pullback_stop_atr, 0.5) * atr)
            self.counters.pullback_entries += 1
            self._arm_pullback_block(BreakoutSide.LONG, bar)
            return self._open_position(
                bar,
                BreakoutSide.LONG,
                bar.close,
                atr,
                note=f"pullback_long er={snapshot.efficiency_ratio:.2f}",
                stop_price=stop,
                style="pullback",
                trail_mult=params.pullback_trail_atr,
            )
        if trend_down and float(bar.high) >= ema_fast - touch:
            if previous is not None and not self._pullback_thrust_ok(bar, atr, BreakoutSide.SHORT):
                return None
            if not self._pullback_wave_ok(bar, atr, BreakoutSide.SHORT):
                return None
            swing_high = max(self._recent_highs) if self._recent_highs else float(bar.high)
            stop = dec(max(swing_high, float(bar.high)) + params.pullback_stop_atr * atr)
            if stop <= bar.close:
                stop = dec(close + max(params.pullback_stop_atr, 0.5) * atr)
            self.counters.pullback_entries += 1
            self._arm_pullback_block(BreakoutSide.SHORT, bar)
            return self._open_position(
                bar,
                BreakoutSide.SHORT,
                bar.close,
                atr,
                note=f"pullback_short er={snapshot.efficiency_ratio:.2f}",
                stop_price=stop,
                style="pullback",
                trail_mult=params.pullback_trail_atr,
            )
        return None

    def _pullback_thrust_ok(self, bar: Bar, atr: float, side: BreakoutSide) -> bool:
        """Подтверждение разворота: пробой экстремума предыдущего бара ИЛИ сильное тело.

        Без этого вход срабатывает на каждом баре медленного дрейфа и превращается
        в серию стопов с полной комиссией (главный источник churn в v4.0).
        """
        params = self._p
        if not params.pullback_require_thrust:
            return True
        previous = self._prev_bar
        close = float(bar.close)
        body = close - float(bar.open)
        sign = 1.0 if side is BreakoutSide.LONG else -1.0
        if previous is not None:
            broke = close > float(previous.high) if side is BreakoutSide.LONG else close < float(previous.low)
            if broke:
                return True
        strong_close = (
            bar.close_position >= 1.0 - 0.35 if side is BreakoutSide.LONG else bar.close_position <= 0.35
        )
        if strong_close and sign * body >= params.pullback_thrust_min_atr * atr:
            return True
        self.counters.thrust_skips += 1
        return False

    def _pullback_wave_ok(self, bar: Bar, atr: float, side: BreakoutSide) -> bool:
        """Волновой кулдаун: не более одного отката на волну в одну сторону."""
        params = self._p
        if self._pullback_block_side is None or self._pullback_block_side is not side:
            return True
        if self._bar_index >= self._pullback_block_until:
            return True
        sign = 1.0 if side is BreakoutSide.LONG else -1.0
        impulse = sign * (float(bar.close) - self._pullback_block_price)
        if impulse >= params.pullback_wave_impulse_atr * atr:
            return True
        self.counters.wave_skips += 1
        return False

    def _arm_pullback_block(self, side: BreakoutSide, bar: Bar) -> None:
        self._pullback_block_side = side
        self._pullback_block_until = self._bar_index + max(int(self._p.pullback_wave_cooldown_bars), 0)
        self._pullback_block_price = float(bar.close)

    # ------------------------------------------------------------------ Engine C
    def _fade_entry(
        self, bar: Bar, snapshot: IndicatorSnapshot, obi: Optional[float]
    ) -> Optional[BreakoutEvent]:
        """Фейд растяжения за полосы Кельтнера с целью на EMA20.

        Полосы считаются по **предыдущему** бару (EMA20/ATR до выноса): иначе сам
        вынос расширяет ATR и EMA, и «отбой» перестаёт определяться.
        """
        params = self._p
        if not self._entry_allowed(bar, snapshot, obi):
            return None
        reference = self._prev_snapshot or snapshot
        atr = reference.atr_points
        center = reference.ema_fast
        if atr <= 0 or center <= 0:
            return None
        upper = center + params.fade_keltner_atr * atr
        lower = center - params.fade_keltner_atr * atr
        close = float(bar.close)
        rub = float(self._meta.rub_per_price_unit)
        fee = float(self._fee_round_trip(bar.close, 1))
        if fee <= 0 or rub <= 0:
            return None
        # Шорт: вынесли выше верхней полосы и закрылись обратно внутрь (отбой).
        if (
            float(bar.high) > upper
            and close < upper
            and bar.close_position <= params.fade_rejection_close_ratio
        ):
            target = dec(center)
            expected = (close - float(target)) * rub
            if expected / fee < params.fade_min_target_fee_ratio:
                return None
            stop = dec(float(bar.high) + params.fade_stop_atr * atr)
            self.counters.fade_entries += 1
            return self._open_position(
                bar,
                BreakoutSide.SHORT,
                bar.close,
                atr,
                note=f"fade_short er={snapshot.efficiency_ratio:.2f}",
                stop_price=stop,
                style="fade",
                target_price=target,
            )
        # Лонг: вынесли ниже нижней полосы и закрылись обратно внутрь.
        if (
            float(bar.low) < lower
            and close > lower
            and bar.close_position >= 1.0 - params.fade_rejection_close_ratio
        ):
            target = dec(center)
            expected = (float(target) - close) * rub
            if expected / fee < params.fade_min_target_fee_ratio:
                return None
            stop = dec(float(bar.low) - params.fade_stop_atr * atr)
            self.counters.fade_entries += 1
            return self._open_position(
                bar,
                BreakoutSide.LONG,
                bar.close,
                atr,
                note=f"fade_long er={snapshot.efficiency_ratio:.2f}",
                stop_price=stop,
                style="fade",
                target_price=target,
            )
        return None

    # ------------------------------------------------------------------ ведение fade
    def _manage(
        self, bar: Bar, snapshot: IndicatorSnapshot
    ) -> List[BreakoutEvent]:
        position = self.position
        if position is None or position.style != "fade":
            return super()._manage(bar, snapshot)
        params = self._p
        events: List[BreakoutEvent] = []
        # 1) Стоп проверяем первым (консервативно), с исполнением по open при гэпе.
        hit = (
            bar.low <= position.stop_price
            if position.side is BreakoutSide.LONG
            else bar.high >= position.stop_price
        )
        if hit:
            fill = position.stop_price
            if position.side is BreakoutSide.LONG and bar.open < position.stop_price:
                fill = bar.open
            elif position.side is BreakoutSide.SHORT and bar.open > position.stop_price:
                fill = bar.open
            events.append(self._close(position, fill, bar.time_ms, BreakoutExitReason.HARD_STOP, bar))
            return events
        # 2) Цель — EMA20 (возврат к средней).
        if position.target_price > 0:
            reached = (
                bar.high >= position.target_price
                if position.side is BreakoutSide.LONG
                else bar.low <= position.target_price
            )
            if reached:
                events.append(
                    self._close(
                        position, position.target_price, bar.time_ms, BreakoutExitReason.TAKE_PROFIT, bar
                    )
                )
                return events
        # 3) Смена режима или тайм-аут: фейд держим не дольше fade_max_bars.
        bars_held = max(
            int(round((bar.time_ms - position.entry_ms) / max(self._bar_minutes * 60_000, 1))), 0
        )
        if params.fade_max_bars > 0 and bars_held >= params.fade_max_bars:
            events.append(self._close(position, bar.close, bar.time_ms, BreakoutExitReason.MAX_HOLD, bar))
            return events
        if snapshot.efficiency_ratio > params.fade_max_er + 0.15:
            events.append(
                self._close(position, bar.close, bar.time_ms, BreakoutExitReason.FAILED_BREAKOUT, bar)
            )
            return events
        # 4) Интрадей-закрытие (если перенос через ночь выключен).
        moment = in_windows(bar.time_ms, self._windows)
        if not params.allow_overnight_hold and not moment:
            events.append(
                self._close(position, bar.close, bar.time_ms, BreakoutExitReason.SESSION_CLOSE, bar)
            )
            return events
        position.update_extremes(bar)
        return events

    # ------------------------------------------------------------------ закрытие
    def _close(
        self,
        position,
        price,
        time_ms: int,
        reason,
        bar: Bar,
    ):
        """После выхода из отката волновой кулдаун начинается заново."""
        if getattr(position, "style", "trend") == "pullback":
            self._arm_pullback_block(position.side, bar)
        return super()._close(position, price, time_ms, reason, bar)


def regime_machine(
    meta, params: BreakoutParams, fee_calculator=None
) -> MomentumBreakoutStateMachine:
    """Фабрика: multi-режим, если он включён в параметрах, иначе обычный пробой."""
    klass = (
        MultiRegimeStateMachine
        if str(params.regime_mode).lower() == "multi"
        else MomentumBreakoutStateMachine
    )
    return klass(meta, params, fee_calculator=fee_calculator)
