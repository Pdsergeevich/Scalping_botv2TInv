"""Сборка общего ядра исполнения для всех режимов."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from bot_a.config import AppConfig
from bot_a.economics.fees import TinkoffFeeCalculator
from bot_a.engine.fsm import PairScalpingStateMachine, StrategyParams
from bot_a.engine.ledger import PortfolioLedger
from bot_a.execution.gateway import OrderExecutionGateway
from bot_a.models.domain import ExecutionTopology, InstrumentMetadata
from bot_a.risk.guard import (
    DailyKillSwitch,
    IntradayEntryWindows,
    MoexSessionSchedule,
    RiskAndScheduleGuard,
)
from bot_a.utils.time_provider import TimeProvider


def create_guard(config: AppConfig, legs: Sequence[InstrumentMetadata]) -> RiskAndScheduleGuard:
    futures = any(meta.is_futures for meta in legs)
    schedule = MoexSessionSchedule(futures=futures, trade_evening_session=config.risk.trade_evening_session)
    kill_switch = DailyKillSwitch(config.risk.max_daily_drawdown_rub)
    return RiskAndScheduleGuard(
        schedule,
        kill_switch,
        config.risk.margin_safety_buffer_pct,
        config.risk.max_position_lots,
        IntradayEntryWindows(config.risk.active_session_windows),
    )


def create_engine(
    legs: Sequence[InstrumentMetadata],
    params: StrategyParams,
    topology: ExecutionTopology,
    gateway: OrderExecutionGateway,
    fee_calculator: TinkoffFeeCalculator,
    clock: TimeProvider,
    guard: RiskAndScheduleGuard,
    ledger: Optional[PortfolioLedger] = None,
    logger: Optional[logging.Logger] = None,
    daily_state_path: Optional[str] = None,
    bootstrap: bool = False,
) -> PairScalpingStateMachine:
    return PairScalpingStateMachine(
        leg1=legs[0],
        leg2=legs[1],
        params=params,
        gateway=gateway,
        fee_calculator=fee_calculator,
        clock=clock,
        guard=guard,
        topology=topology,
        ledger=ledger,
        logger=logger,
        daily_state_path=daily_state_path,
        bootstrap=bootstrap,
    )
