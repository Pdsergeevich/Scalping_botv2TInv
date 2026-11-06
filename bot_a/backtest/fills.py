"""Общие примитивы исполнения: рублёвый номинал, вход/выход с проскальзыванием.

Вынесено из бэктестера, чтобы живой свинг-раннер и бэктест применяли **одни и те же**
правила филла (иначе paper и backtest разъезжаются по деньгам).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Tuple

from bot_a.engine.breakout_fsm import BreakoutParams, BreakoutSide
from bot_a.models.domain import InstrumentMetadata
from bot_a.utils.quotation import dec


def notional_rub(meta: InstrumentMetadata, price: Decimal, quantity: int) -> Decimal:
    """Рублёвый номинал позиции (для комиссии брокера)."""
    return meta.value_in_rub(price) * Decimal(max(int(quantity), 1))


def slippage_steps(meta: InstrumentMetadata, params: BreakoutParams) -> Decimal:
    return meta.min_price_increment * Decimal(max(int(params.slippage_steps), 0))


def apply_entry_fill(
    meta: InstrumentMetadata,
    params: BreakoutParams,
    price: Decimal,
    side: BreakoutSide,
    stop_points: float,
) -> Tuple[Decimal, Decimal]:
    """Цена входа с проскальзыванием и стоп, пересчитанный от фактической цены."""
    slip = slippage_steps(meta, params)
    distance = dec(stop_points)
    if side is BreakoutSide.LONG:
        fill = price + slip
        return fill, fill - distance
    fill = price - slip
    return fill, fill + distance


def apply_exit_fill(
    meta: InstrumentMetadata, params: BreakoutParams, price: Decimal, side: BreakoutSide
) -> Decimal:
    slip = slippage_steps(meta, params)
    return price - slip if side is BreakoutSide.LONG else price + slip
