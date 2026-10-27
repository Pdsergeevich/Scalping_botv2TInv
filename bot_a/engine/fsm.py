"""Конечный автомат парного скальпинга (единое ядро для всех режимов).

Движок не знает, откуда пришёл стакан (gRPC или реплей) и куда уходит заявка
(биржа или симулятор): он работает через ``OrderExecutionGateway``,
``TimeProvider`` и ``RiskAndScheduleGuard``.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from bot_a.economics.fees import TinkoffFeeCalculator
from bot_a.engine.ledger import PortfolioLedger
from bot_a.execution.gateway import (
    ExecutionEvent,
    ExecutionEventKind,
    OrderExecutionGateway,
    OrderRequest,
)
from bot_a.math_engine.pair_spread import PairSpreadCalculator
from bot_a.models.domain import (
    ExecutionTopology,
    ExitReason,
    FsmState,
    InstrumentMetadata,
    OpenPosition,
    OrderBookSnapshot,
    OrderType,
    Side,
    SpreadMetrics,
    TradePrint,
    TradeRecord,
)
from bot_a.risk.guard import RiskAndScheduleGuard
from bot_a.utils.quotation import ceil_to_increment, dec, round_to_increment, shift_by_steps
from bot_a.utils.time_provider import TimeProvider, ms_to_msk

ZERO = Decimal("0")

ROLE_ENTRY_LEG1 = "entry1"
ROLE_HEDGE_LEG2 = "hedge2"
ROLE_EXIT_LEG1 = "exit1"
ROLE_EXIT_LEG1_MARKET = "exit1_market"
ROLE_EXIT_LEG2 = "hedge2_exit"
ROLE_FLATTEN_LEG1 = "flatten1"
ROLE_FLATTEN_LEG2 = "flatten2"
ROLE_TP_LEG1 = "tp1"

_LEG1_CLOSE_ROLES = (ROLE_EXIT_LEG1, ROLE_EXIT_LEG1_MARKET, ROLE_FLATTEN_LEG1, ROLE_TP_LEG1)
_LEG2_CLOSE_ROLES = (ROLE_EXIT_LEG2, ROLE_FLATTEN_LEG2)


@dataclass(frozen=True, slots=True)
class StrategyParams:
    """Параметры стратегии (полностью сериализуемы для оптимизатора)."""

    z_entry: float = 2.0
    z_exit: float = 0.2
    z_cancel: float = 1.0
    z_stop: float = 4.2
    ewma_window: int = 400
    obi_depth: int = 5
    obi_entry_guard: float = 0.45
    obi_toxic_guard: float = 0.70
    requote_step_threshold: int = 1
    requote_cooldown_ms: int = 250
    requote_z_slack: float = 0.15
    max_hold_seconds: int = 600
    max_exit_quote_seconds: int = 12
    time_stop_quote_seconds: int = 30
    cooldown_seconds: int = 3
    quantity_lots: int = 1
    min_edge_multiplier: Decimal = Decimal("1.8")
    slippage_buffer_steps: int = 1
    beta: float = 1.0
    hedge_timeout_ms: int = 3000
    flatten_retry_ms: int = 2000
    # --- v2: экономические исправления ---
    use_frozen_mu: bool = True
    use_profit_lock: bool = True
    beta_mode: str = "auto"
    min_net_profit_multiplier: Decimal = Decimal("1.35")
    profit_floor_multiplier: Decimal = Decimal("1.25")
    dual_profit_floor_multiplier: Decimal = Decimal("1.3")
    tp_deviation_fraction: float = 0.55
    ghost_reversion_z: float = 0.3
    ghost_flat_seconds: int = 45
    scratch_seconds: int = 60
    beta_min: float = 0.1
    beta_max: float = 20.0
    beta_min_samples: int = 30
    use_entry_z_cap: bool = True
    entry_z_cap_slack: float = 1.0
    # --- v2.2: Z-peak хук, breakeven-ratchet, режимный фильтр направления ---
    use_z_peak_hook: bool = True
    z_hook_pullback: float = 0.12
    z_min_trigger: float = 0.15
    z_peak_reset_gap: float = 0.25
    use_breakeven_ratchet: bool = False
    ratchet_arm_multiplier: Decimal = Decimal("2.2")
    ratchet_arm_target_fraction: float = 0.90
    ratchet_arm_step_buffer: Decimal = Decimal("1")
    ratchet_retrace_steps: int = 2
    ratchet_min_hold_seconds: int = 120
    ratchet_exit_multiplier: Decimal = Decimal("1.15")
    ratchet_exit_target_fraction: float = 0.60
    use_tp_decay: bool = False
    tp_decay_start_fraction: float = 0.5
    tp_decay_floor_multiplier: Decimal = Decimal("1.15")
    tp_min_fee_multiple: Decimal = Decimal("2.5")
    use_regime_filter: bool = True
    regime_window: int = 12
    regime_dominant_share: float = 0.65
    regime_stable_share: float = 0.35
    # --- v2.3: ценовой стоп по Leg1 для SingleLegLeadLag ---
    use_single_leg_price_stop: bool = True
    single_leg_stop_to_target_ratio: float = 1.5
    single_leg_stop_min_steps: int = 4
    lead_failure_z: float = 0.5
    lead_failure_seconds: int = 60
    # --- v2.4: дневной предохранитель, макро-дрейф беты, фильтр волатильность/комиссия ---
    use_daily_stop_breaker: bool = True
    max_daily_hard_stops: int = 1
    max_daily_loss_fee_multiple: Decimal = Decimal("3.5")
    use_macro_beta_drift_guard: bool = True
    macro_beta_window: int = 14400
    macro_beta_min_samples: int = 1800
    max_beta_drift: float = 0.008
    use_vol_to_fee_filter: bool = True
    min_sigma_to_fee_ratio: float = 1.15
    # --- v2.4.1: квантование статистических окон по времени (L2 100–200 мс) ---
    stat_sample_interval_seconds: float = 0.0
    # --- v2.5: подтверждение сигнала двумя подряд статистическими замерами ---
    use_entry_persistence: bool = False
    entry_confirm_samples: int = 2

    def with_overrides(self, **overrides: Any) -> "StrategyParams":
        return replace(self, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zEntry": self.z_entry,
            "zExit": self.z_exit,
            "zCancel": self.z_cancel,
            "zStop": self.z_stop,
            "ewmaWindow": self.ewma_window,
            "obiDepth": self.obi_depth,
            "obiEntryGuard": self.obi_entry_guard,
            "obiToxicGuard": self.obi_toxic_guard,
            "requoteStepThreshold": self.requote_step_threshold,
            "requoteCooldownMs": self.requote_cooldown_ms,
            "maxHoldSeconds": self.max_hold_seconds,
            "maxExitQuoteSeconds": self.max_exit_quote_seconds,
            "cooldownSeconds": self.cooldown_seconds,
            "quantityLots": self.quantity_lots,
            "minEdgeMultiplier": str(self.min_edge_multiplier),
            "slippageBufferSteps": self.slippage_buffer_steps,
            "beta": self.beta,
            "useFrozenMu": self.use_frozen_mu,
            "useProfitLock": self.use_profit_lock,
            "betaMode": self.beta_mode,
            "minNetProfitMultiplier": str(self.min_net_profit_multiplier),
            "profitFloorMultiplier": str(self.profit_floor_multiplier),
            "dualProfitFloorMultiplier": str(self.dual_profit_floor_multiplier),
            "tpDeviationFraction": self.tp_deviation_fraction,
            "ghostReversionZ": self.ghost_reversion_z,
            "ghostFlatSeconds": self.ghost_flat_seconds,
            "scratchSeconds": self.scratch_seconds,
            "timeStopQuoteSeconds": self.time_stop_quote_seconds,
            "betaMin": self.beta_min,
            "betaMax": self.beta_max,
            "useEntryZCap": self.use_entry_z_cap,
            "entryZCapSlack": self.entry_z_cap_slack,
            "useZPeakHook": self.use_z_peak_hook,
            "zHookPullback": self.z_hook_pullback,
            "zMinTrigger": self.z_min_trigger,
            "zPeakResetGap": self.z_peak_reset_gap,
            "useBreakevenRatchet": self.use_breakeven_ratchet,
            "ratchetArmMultiplier": str(self.ratchet_arm_multiplier),
            "ratchetArmTargetFraction": self.ratchet_arm_target_fraction,
            "ratchetArmStepBuffer": str(self.ratchet_arm_step_buffer),
            "ratchetRetraceSteps": self.ratchet_retrace_steps,
            "ratchetMinHoldSeconds": self.ratchet_min_hold_seconds,
            "ratchetExitMultiplier": str(self.ratchet_exit_multiplier),
            "ratchetExitTargetFraction": self.ratchet_exit_target_fraction,
            "tpMinFeeMultiple": str(self.tp_min_fee_multiple),
            "useSingleLegPriceStop": self.use_single_leg_price_stop,
            "singleLegStopToTargetRatio": self.single_leg_stop_to_target_ratio,
            "singleLegStopMinSteps": self.single_leg_stop_min_steps,
            "leadFailureZ": self.lead_failure_z,
            "leadFailureSeconds": self.lead_failure_seconds,
            "useTpDecay": self.use_tp_decay,
            "tpDecayStartFraction": self.tp_decay_start_fraction,
            "tpDecayFloorMultiplier": str(self.tp_decay_floor_multiplier),
            "useRegimeFilter": self.use_regime_filter,
            "regimeWindow": self.regime_window,
            "regimeDominantShare": self.regime_dominant_share,
            "regimeStableShare": self.regime_stable_share,
            "useDailyStopBreaker": self.use_daily_stop_breaker,
            "maxDailyHardStops": self.max_daily_hard_stops,
            "maxDailyLossFeeMultiple": str(self.max_daily_loss_fee_multiple),
            "useMacroBetaDriftGuard": self.use_macro_beta_drift_guard,
            "macroBetaWindow": self.macro_beta_window,
            "maxBetaDrift": self.max_beta_drift,
            "useVolToFeeFilter": self.use_vol_to_fee_filter,
            "minSigmaToFeeRatio": self.min_sigma_to_fee_ratio,
            "statSampleIntervalSeconds": self.stat_sample_interval_seconds,
            "useEntryPersistence": self.use_entry_persistence,
            "entryConfirmSamples": self.entry_confirm_samples,
        }

    @staticmethod
    def from_mapping(data: Dict[str, Any]) -> "StrategyParams":
        """Собирает параметры из секции ``strategy`` appsettings.json."""
        defaults = StrategyParams()
        beta_mode = str(data.get("betaMode", ""))
        if not beta_mode or beta_mode == "auto":
            # v2.5: "auto" раньше молча превращался в "same_underlying" внутри
            # PairSpreadCalculator, из-за чего бэктест и live считали бету по-разному.
            # Теперь "auto" означает ту же скользящую оценку, что задают пресеты.
            legacy = data.get("useDynamicBeta")
            if legacy is None:
                beta_mode = "slow_ratio"
            else:
                beta_mode = "dynamic" if bool(legacy) else "static"
        return StrategyParams(
            z_entry=float(data.get("zEntry", defaults.z_entry)),
            z_exit=float(data.get("zExit", defaults.z_exit)),
            z_cancel=float(data.get("zCancel", defaults.z_cancel)),
            z_stop=float(data.get("zStop", defaults.z_stop)),
            ewma_window=int(data.get("ewmaWindow", defaults.ewma_window)),
            obi_depth=int(data.get("obiDepth", defaults.obi_depth)),
            obi_entry_guard=float(data.get("obiEntryGuard", defaults.obi_entry_guard)),
            obi_toxic_guard=float(data.get("obiToxicGuard", defaults.obi_toxic_guard)),
            requote_step_threshold=int(data.get("requoteStepThreshold", defaults.requote_step_threshold)),
            requote_cooldown_ms=int(data.get("requoteCooldownMs", defaults.requote_cooldown_ms)),
            requote_z_slack=float(data.get("requoteZSlack", defaults.requote_z_slack)),
            max_hold_seconds=int(data.get("maxHoldSeconds", defaults.max_hold_seconds)),
            max_exit_quote_seconds=int(data.get("maxExitQuoteSeconds", defaults.max_exit_quote_seconds)),
            time_stop_quote_seconds=int(data.get("timeStopQuoteSeconds", defaults.time_stop_quote_seconds)),
            cooldown_seconds=int(data.get("cooldownSeconds", defaults.cooldown_seconds)),
            quantity_lots=int(data.get("quantityLots", defaults.quantity_lots)),
            min_edge_multiplier=dec(data.get("minEdgeMultiplier", defaults.min_edge_multiplier)),
            slippage_buffer_steps=int(data.get("slippageBufferSteps", defaults.slippage_buffer_steps)),
            beta=float(data.get("beta", defaults.beta)),
            hedge_timeout_ms=int(data.get("hedgeTimeoutMs", defaults.hedge_timeout_ms)),
            flatten_retry_ms=int(data.get("flattenRetryMs", defaults.flatten_retry_ms)),
            use_frozen_mu=bool(data.get("useFrozenMu", defaults.use_frozen_mu)),
            use_profit_lock=bool(data.get("useProfitLock", defaults.use_profit_lock)),
            beta_mode=beta_mode,
            min_net_profit_multiplier=dec(data.get("minNetProfitMultiplier", defaults.min_net_profit_multiplier)),
            profit_floor_multiplier=dec(data.get("profitFloorMultiplier", defaults.profit_floor_multiplier)),
            dual_profit_floor_multiplier=dec(data.get("dualProfitFloorMultiplier", defaults.dual_profit_floor_multiplier)),
            tp_deviation_fraction=float(data.get("tpDeviationFraction", defaults.tp_deviation_fraction)),
            ghost_reversion_z=float(data.get("ghostReversionZ", defaults.ghost_reversion_z)),
            ghost_flat_seconds=int(data.get("ghostFlatSeconds", defaults.ghost_flat_seconds)),
            scratch_seconds=int(data.get("scratchSeconds", defaults.scratch_seconds)),
            beta_min=float(data.get("betaMin", defaults.beta_min)),
            beta_max=float(data.get("betaMax", defaults.beta_max)),
            beta_min_samples=int(data.get("betaMinSamples", defaults.beta_min_samples)),
            use_entry_z_cap=bool(data.get("useEntryZCap", defaults.use_entry_z_cap)),
            entry_z_cap_slack=float(data.get("entryZCapSlack", defaults.entry_z_cap_slack)),
            use_z_peak_hook=bool(data.get("useZPeakHook", defaults.use_z_peak_hook)),
            z_hook_pullback=float(data.get("zHookPullback", defaults.z_hook_pullback)),
            z_min_trigger=float(data.get("zMinTrigger", defaults.z_min_trigger)),
            z_peak_reset_gap=float(data.get("zPeakResetGap", defaults.z_peak_reset_gap)),
            use_breakeven_ratchet=bool(data.get("useBreakevenRatchet", defaults.use_breakeven_ratchet)),
            ratchet_arm_multiplier=dec(data.get("ratchetArmMultiplier", defaults.ratchet_arm_multiplier)),
            ratchet_arm_target_fraction=float(
                data.get("ratchetArmTargetFraction", defaults.ratchet_arm_target_fraction)
            ),
            ratchet_arm_step_buffer=dec(data.get("ratchetArmStepBuffer", defaults.ratchet_arm_step_buffer)),
            ratchet_retrace_steps=int(data.get("ratchetRetraceSteps", defaults.ratchet_retrace_steps)),
            ratchet_min_hold_seconds=int(
                data.get("ratchetMinHoldSeconds", defaults.ratchet_min_hold_seconds)
            ),
            ratchet_exit_multiplier=dec(data.get("ratchetExitMultiplier", defaults.ratchet_exit_multiplier)),
            ratchet_exit_target_fraction=float(
                data.get("ratchetExitTargetFraction", defaults.ratchet_exit_target_fraction)
            ),
            tp_min_fee_multiple=dec(data.get("tpMinFeeMultiple", defaults.tp_min_fee_multiple)),
            use_single_leg_price_stop=bool(
                data.get("useSingleLegPriceStop", defaults.use_single_leg_price_stop)
            ),
            single_leg_stop_to_target_ratio=float(
                data.get("singleLegStopToTargetRatio", defaults.single_leg_stop_to_target_ratio)
            ),
            single_leg_stop_min_steps=int(
                data.get("singleLegStopMinSteps", defaults.single_leg_stop_min_steps)
            ),
            lead_failure_z=float(data.get("leadFailureZ", defaults.lead_failure_z)),
            lead_failure_seconds=int(data.get("leadFailureSeconds", defaults.lead_failure_seconds)),
            use_tp_decay=bool(data.get("useTpDecay", defaults.use_tp_decay)),
            tp_decay_start_fraction=float(data.get("tpDecayStartFraction", defaults.tp_decay_start_fraction)),
            tp_decay_floor_multiplier=dec(data.get("tpDecayFloorMultiplier", defaults.tp_decay_floor_multiplier)),
            use_regime_filter=bool(data.get("useRegimeFilter", defaults.use_regime_filter)),
            regime_window=int(data.get("regimeWindow", defaults.regime_window)),
            regime_dominant_share=float(data.get("regimeDominantShare", defaults.regime_dominant_share)),
            regime_stable_share=float(data.get("regimeStableShare", defaults.regime_stable_share)),
            use_daily_stop_breaker=bool(data.get("useDailyStopBreaker", defaults.use_daily_stop_breaker)),
            max_daily_hard_stops=int(data.get("maxDailyHardStops", defaults.max_daily_hard_stops)),
            max_daily_loss_fee_multiple=dec(
                data.get("maxDailyLossFeeMultiple", defaults.max_daily_loss_fee_multiple)
            ),
            use_macro_beta_drift_guard=bool(
                data.get("useMacroBetaDriftGuard", defaults.use_macro_beta_drift_guard)
            ),
            macro_beta_window=int(data.get("macroBetaWindow", defaults.macro_beta_window)),
            macro_beta_min_samples=int(
                data.get("macroBetaMinSamples", defaults.macro_beta_min_samples)
            ),
            max_beta_drift=float(data.get("maxBetaDrift", defaults.max_beta_drift)),
            use_vol_to_fee_filter=bool(data.get("useVolToFeeFilter", defaults.use_vol_to_fee_filter)),
            min_sigma_to_fee_ratio=float(
                data.get("minSigmaToFeeRatio", defaults.min_sigma_to_fee_ratio)
            ),
            stat_sample_interval_seconds=float(
                data.get("statSampleIntervalSeconds", defaults.stat_sample_interval_seconds)
            ),
            use_entry_persistence=bool(
                data.get("useEntryPersistence", defaults.use_entry_persistence)
            ),
            entry_confirm_samples=max(
                int(data.get("entryConfirmSamples", defaults.entry_confirm_samples)), 1
            ),
        )


def _blend_price(old_price: Optional[Decimal], old_quantity: int, new_price: Decimal, new_quantity: int) -> Decimal:
    if old_price is None or old_quantity <= 0:
        return new_price
    total = old_quantity + new_quantity
    if total <= 0:
        return new_price
    return (old_price * Decimal(old_quantity) + new_price * Decimal(new_quantity)) / Decimal(total)


class PairScalpingStateMachine:
    """FSM исполнения пары: прогрев -> вход -> хедж -> позиция -> выход."""

    def __init__(
        self,
        leg1: InstrumentMetadata,
        leg2: InstrumentMetadata,
        params: StrategyParams,
        gateway: OrderExecutionGateway,
        fee_calculator: TinkoffFeeCalculator,
        clock: TimeProvider,
        guard: RiskAndScheduleGuard,
        topology: ExecutionTopology = ExecutionTopology.SINGLE_LEG_LEAD_LAG,
        ledger: Optional[PortfolioLedger] = None,
        logger: Optional[logging.Logger] = None,
        daily_state_path: Optional[Any] = None,
        bootstrap: bool = False,
    ) -> None:
        self._leg1 = leg1
        self._leg2 = leg2
        self._params = params
        self._gateway = gateway
        self._fees = fee_calculator
        self._clock = clock
        self._guard = guard
        self._topology = topology
        self._ledger = ledger if ledger is not None else PortfolioLedger()
        self._log = logger or logging.getLogger("bot_a.engine")
        self._calc = PairSpreadCalculator(
            leg1=leg1,
            leg2=leg2,
            beta=params.beta,
            ewma_window=params.ewma_window,
            obi_depth=params.obi_depth,
            beta_mode=params.beta_mode,
            beta_min=params.beta_min,
            beta_max=params.beta_max,
            beta_min_samples=params.beta_min_samples,
            macro_beta_window=params.macro_beta_window,
            macro_beta_min_samples=params.macro_beta_min_samples,
            stat_sample_interval_seconds=params.stat_sample_interval_seconds,
        )
        self._state = FsmState.WARMUP
        self._state_since_ms = clock.now_ms()
        self._last_metrics: Optional[SpreadMetrics] = None
        self._position: Optional[OpenPosition] = None
        self._exit_reason: ExitReason = ExitReason.TAKE_PROFIT
        self._trade_seq = 0
        self._last_requote_ms = 0
        self._last_exit_requote_ms = 0
        self._cancel_requested_role: Optional[str] = None
        self._roles: Dict[str, str] = {}
        self._role_by_order: Dict[str, str] = {}
        self._log_throttle: Dict[str, int] = {}
        # --- v2.2: пик Z и режимный буфер рублёвых стоимостей ног ---
        self._z_peak = 0.0
        self._z_prev = 0.0
        self._z_pullback_streak = 0
        # --- v2.5: подтверждение сигнала двумя подряд статистическими замерами ---
        self._entry_confirm_streak = 0
        self._entry_signal_since_ms: Optional[int] = None
        regime_len = max(int(params.regime_window), 2) + 1
        self._regime_v1: Deque[float] = deque(maxlen=regime_len)
        self._regime_v2: Deque[float] = deque(maxlen=regime_len)
        # --- v2.4: дневной предохранитель пары ---
        self._daily_key = ""
        self._daily_hard_stops = 0
        self._daily_net_rub: Decimal = ZERO
        self._daily_locked_reason: Optional[str] = None
        # --- v2.4.1: персистентность дневного состояния и bootstrap-прогрев ---
        self._daily_state_path = Path(daily_state_path) if daily_state_path else None
        self._bootstrap = bool(bootstrap)
        # Подхват дневного состояния сразу при старте (до первого тика).
        self._roll_daily(clock.now_ms())
        self._counters: Dict[str, int] = {
            "signals": 0,
            "edgeSkips": 0,
            "guardSkips": 0,
            "overextendedSkips": 0,
            "hookSkips": 0,
            "regimeSkips": 0,
            "sessionSkips": 0,
            "dailyStopSkips": 0,
            "betaDriftSkips": 0,
            "volFilterSkips": 0,
            "persistenceSkips": 0,
            "ratchetExits": 0,
            "entries": 0,
            "rejects": 0,
            "trades": 0,
        }

    # ------------------------------------------------------------------ свойства
    @property
    def state(self) -> FsmState:
        return self._state

    @property
    def metrics(self) -> Optional[SpreadMetrics]:
        return self._last_metrics

    @property
    def position(self) -> Optional[OpenPosition]:
        return self._position

    @property
    def ledger(self) -> PortfolioLedger:
        return self._ledger

    @property
    def gateway(self) -> OrderExecutionGateway:
        return self._gateway

    @property
    def topology(self) -> ExecutionTopology:
        return self._topology

    @property
    def params(self) -> StrategyParams:
        return self._params

    @property
    def counters(self) -> Dict[str, int]:
        return dict(self._counters)

    # ------------------------------------------------------------------ входы событий
    async def on_order_book(self, leg: int, book: OrderBookSnapshot) -> None:
        self._gateway.on_order_book(leg, book)
        metrics = self._calc.on_book(leg - 1, book)
        await self._drain_events()
        if metrics is not None:
            self._last_metrics = metrics
            await self._on_metrics(metrics)
        await self._drain_events()

    async def on_trade(self, leg: int, trade: TradePrint) -> None:
        self._gateway.on_trade(leg, trade)
        await self._drain_events()

    async def on_clock(self) -> None:
        await self._gateway.advance_clock()
        await self._drain_events()
        await self._on_time()

    async def shutdown(self, flatten: bool = True) -> None:
        """Аккуратное завершение: снятие заявок и (опционально) закрытие позиции."""
        await self._gateway.cancel_all()
        await self._drain_events()
        if flatten and self._position is not None and self._position.leg1_filled_qty > self._position.leg1_closed_qty:
            await self._emergency_flatten(ExitReason.SHUTDOWN, "завершение работы")
            await self._gateway.advance_clock()
            await self._drain_events()

    def snapshot(self) -> Dict[str, Any]:
        position = self._position
        return {
            "state": self._state.value,
            "topology": self._topology.value,
            "positionOpen": position is not None,
            "positionLeg1Qty": position.leg1_filled_qty if position else 0,
            "zscore": self._last_metrics.zscore if self._last_metrics else 0.0,
            "trades": self._ledger.trade_count,
            "netPnlRub": str(self._ledger.net_pnl_rub),
            "feesRub": str(self._ledger.fees_rub),
            "dailyLockedReason": self._daily_locked_reason,
            "dailyNetPnlRub": str(self._daily_net_rub),
            "dailyHardStops": self._daily_hard_stops,
            "betaDrift": self._last_metrics.beta_drift if self._last_metrics else None,
            "counters": dict(self._counters),
            "fees": self._fees.snapshot(),
        }

    # ------------------------------------------------------------------ переходы
    def _transition(self, new_state: FsmState, reason: str) -> None:
        if new_state is self._state:
            return
        self._log.info(
            "FSM %s -> %s | %s",
            self._state.value,
            new_state.value,
            reason,
        )
        self._state = new_state
        self._state_since_ms = self._clock.now_ms()

    def _log_throttled(self, key: str, level: int, message: str, *args: Any, interval_ms: int = 5000) -> None:
        now = self._clock.now_ms()
        last = self._log_throttle.get(key, -10**12)
        if now - last < interval_ms:
            return
        self._log_throttle[key] = now
        self._log.log(level, message, *args)

    def _register_order(self, role: str, order_id: str) -> None:
        self._roles[role] = order_id
        self._role_by_order[order_id] = role

    def _release_role(self, role: str) -> Optional[str]:
        order_id = self._roles.pop(role, None)
        if order_id is not None:
            self._role_by_order.pop(order_id, None)
        return order_id

    # ------------------------------------------------------------------ обработка событий шлюза
    async def _drain_events(self) -> None:
        while True:
            event = self._gateway.events.try_read()
            if event is None:
                return
            await self._handle_event(event)

    async def _handle_event(self, event: ExecutionEvent) -> None:
        if event.kind is ExecutionEventKind.FILL:
            await self._handle_fill(event)
        elif event.kind is ExecutionEventKind.CANCELLED:
            await self._handle_cancelled(event)
        elif event.kind is ExecutionEventKind.REJECTED:
            await self._handle_rejected(event)

    async def _handle_cancelled(self, event: ExecutionEvent) -> None:
        role = self._role_by_order.get(event.order_id)
        if role is None:
            return
        self._release_role(role)
        if role == ROLE_ENTRY_LEG1 and self._state is FsmState.QUOTING_ENTRY_LEG1:
            self._cancel_requested_role = None
            if self._position is not None and self._position.leg1_filled_qty == 0:
                self._position = None
            self._transition(FsmState.IDLE, "входная котировка снята")
        elif role == ROLE_HEDGE_LEG2 and self._state is FsmState.HEDGING_ENTRY_LEG2:
            await self._hedge_entry("IOC-хедж не исполнен — переходим в маркет")
        elif role == ROLE_EXIT_LEG1_MARKET and self._state is FsmState.QUOTING_EXIT_LEG1:
            # Рыночная заявка не исполнилась — разрешаем повторную попытку.
            if self._position is not None:
                self._position.tags.pop("exitMarketSent", None)

    async def _handle_rejected(self, event: ExecutionEvent) -> None:
        role = self._role_by_order.get(event.order_id)
        self._counters["rejects"] += 1
        self._log.warning("Заявка %s отклонена: %s", event.order_id, event.message)
        if role is None:
            return
        self._release_role(role)
        if role == ROLE_ENTRY_LEG1 and self._state is FsmState.QUOTING_ENTRY_LEG1:
            self._position = None
            self._transition(FsmState.IDLE, "входная заявка отклонена")
        elif role == ROLE_HEDGE_LEG2 and self._state is FsmState.HEDGING_ENTRY_LEG2:
            await self._hedge_entry("хедж отклонён — повтор рынком")
        elif role == ROLE_TP_LEG1:
            # Отказ прибыльного тейка не критичен: остаются Z-выход, скретч и time-stop.
            self._log.warning("Прибыльный тейк-профит отклонён, продолжаем вести позицию")
        elif role in _LEG1_CLOSE_ROLES or role in _LEG2_CLOSE_ROLES:
            await self._emergency_flatten(ExitReason.EMERGENCY, f"заявка {role} отклонена брокером")

    # ------------------------------------------------------------------ метрики
    async def _on_metrics(self, metrics: SpreadMetrics) -> None:
        if self._state is FsmState.WARMUP:
            if metrics.is_warm:
                self._transition(FsmState.IDLE, f"прогрев завершён: {metrics.tick_count} тиков")
            return
        if self._state is FsmState.IDLE:
            await self._maybe_enter(metrics)
        elif self._state is FsmState.QUOTING_ENTRY_LEG1:
            await self._manage_entry_quote(metrics)
        elif self._state is FsmState.IN_POSITION:
            await self._manage_position(metrics)
        elif self._state is FsmState.QUOTING_EXIT_LEG1:
            await self._manage_exit_quote(metrics)

    # ------------------------------------------------------------------ v2.2: фильтры входа
    def _update_z_peak(self, z: float, allow_arm: bool = True) -> None:
        """Отслеживает локальный пик |Z| и разворот направления (инфлексию).

        ``allow_arm=False`` (v2.5): одиночный выброс не армирует пик, но откат и
        сброс продолжают обновляться, чтобы состояние хука оставалось корректным.
        """
        params = self._params
        az = abs(z)
        if az < params.z_entry - max(params.z_peak_reset_gap, 0.0):
            self._z_peak = 0.0
            self._z_pullback_streak = 0
        else:
            if az >= params.z_entry and az > self._z_peak and allow_arm:
                self._z_peak = az
            if (z >= 0 and z < self._z_prev) or (z < 0 and z > self._z_prev):
                self._z_pullback_streak += 1
            else:
                self._z_pullback_streak = 0
        self._z_prev = z

    def _z_peak_confirmed(self, z: float) -> bool:
        """Вход разрешён только после отката от пика (истощение импульса)."""
        params = self._params
        az = abs(z)
        if self._z_peak < params.z_entry:
            return False
        if self._z_peak - az >= max(params.z_hook_pullback, 0.0):
            return True
        min_trigger = params.z_entry - max(params.z_min_trigger, 0.0)
        return self._z_pullback_streak >= 2 and az >= min_trigger

    def _update_regime(self, metrics: SpreadMetrics) -> None:
        """Копит рублёвые стоимости ног для оценки направления движения.

        v2.4.1: буфер пополняется только на статистических сэмплах, чтобы окно
        ``K`` измерялось в секундах, а не в пакетах L2.
        """
        if not self._params.use_regime_filter:
            return
        if not getattr(metrics, "stat_sampled", True):
            return
        self._regime_v1.append(float(metrics.micro_leg1) * float(self._leg1.rub_per_price_unit))
        self._regime_v2.append(float(metrics.micro_leg2) * float(self._leg2.rub_per_price_unit))

    def _regime_confirmed(self, z: float) -> bool:
        """Проверяет, что нога 1 действительно должна вернуться к ноге 2.

        Допустимы два режима:
        * **A (отстающая нога)**: нога 2 дала не менее `regime_dominant_share`
          движения спреда, нога 1 почти не сдвинулась — входим в направлении ноги 2;
        * **B (изолированный выброс)**: нога 1 сдвинулась в одиночку при стабильной
          ноге 2 — гасим выброс ноги 1.

        Если обе ноги уходят в одну сторону сопоставимым темпом (тренд), вход блокируется.
        """
        params = self._params
        if not params.use_regime_filter or self._topology is not ExecutionTopology.SINGLE_LEG_LEAD_LAG:
            return True
        if len(self._regime_v1) < (params.regime_window + 1):
            return True
        delta1 = self._regime_v1[-1] - self._regime_v1[0]
        delta2 = self._regime_v2[-1] - self._regime_v2[0]
        move1, move2 = abs(delta1), abs(delta2)
        if move1 <= 1e-9 and move2 <= 1e-9:
            return True
        share2 = move2 / (move1 + move2)
        if (
            share2 >= params.regime_dominant_share
            and move1 <= params.regime_stable_share * move2
            and delta2 * z > 0
        ):
            return True
        if (
            move2 <= params.regime_stable_share * move1
            and delta1 * z < 0
        ):
            return True
        return False

    # ------------------------------------------------------------------ v2.4: риск-режимы
    @staticmethod
    def _sigma_leg1_rub(metrics: SpreadMetrics) -> Decimal:
        """σ спреда, приведённая к рублёвой стоимости Leg1 (делим на бету)."""
        beta = metrics.beta if metrics.beta > 1e-9 else 1.0
        return dec(metrics.sigma_rub / beta)

    def _roll_daily(self, now_ms: int) -> None:
        """Сбрасывает дневные счётчики при смене торговой даты (MSK)."""
        key = ms_to_msk(now_ms).strftime("%Y-%m-%d")
        if key == self._daily_key:
            return
        self._daily_key = key
        self._daily_hard_stops = 0
        self._daily_net_rub = ZERO
        self._daily_locked_reason = None
        self._load_daily_state()

    # -------------------------------------------------- v2.4.1: персистентный стейт дня
    def _load_daily_state(self) -> None:
        """Подхватывает состояние дня из файла (только если дата совпадает с текущей)."""
        path = self._daily_state_path
        if path is None or not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._log.warning("Состояние дня не прочитано (%s): %s", path, exc)
            return
        if str(payload.get("date", "")) != self._daily_key:
            return
        self._daily_hard_stops = int(payload.get("daily_hard_stops", 0) or 0)
        self._daily_net_rub = dec(payload.get("daily_net_rub", 0))
        reason = payload.get("daily_locked_reason")
        self._daily_locked_reason = str(reason) if reason else None
        self._log.info(
            "Состояние дня восстановлено: date=%s, HardStop=%d, net=%s RUB, блокировка=%s",
            self._daily_key,
            self._daily_hard_stops,
            self._daily_net_rub,
            self._daily_locked_reason or "нет",
        )

    def _save_daily_state(self) -> None:
        path = self._daily_state_path
        if path is None:
            return
        payload = {
            "date": self._daily_key,
            "daily_hard_stops": self._daily_hard_stops,
            "daily_net_rub": float(self._daily_net_rub),
            "daily_locked_reason": self._daily_locked_reason or "",
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            self._log.warning("Не удалось сохранить состояние дня (%s): %s", path, exc)

    @property
    def bootstrap(self) -> bool:
        return self._bootstrap

    def set_bootstrap(self, value: bool) -> None:
        self._bootstrap = bool(value)

    def spread_state(self) -> Dict[str, Any]:
        """Снимок статистики спреда (прогрев, mu/sigma/беты) для логов и проверок."""
        metrics = self._last_metrics
        drift = metrics.beta_drift if metrics is not None else None
        return {
            "state": self._state.value,
            "tickCount": self._calc.ewma.count,
            "isWarm": self._calc.ewma.is_warm,
            "mu": metrics.mean_rub if metrics is not None else 0.0,
            "spreadRub": metrics.spread_rub if metrics is not None else 0.0,
            "sigmaRub": metrics.sigma_rub if metrics is not None else 0.0,
            "sigmaLeg1Rub": (
                float(self._sigma_leg1_rub(metrics)) if metrics is not None else 0.0
            ),
            "betaSlow": self._calc.beta,
            "betaMacro": self._calc.beta_macro,
            "betaDriftPct": (drift * 100.0) if drift is not None else None,
            "regimeSamples": len(self._regime_v1),
            "regimeCapacity": self._regime_v1.maxlen,
        }

    @property
    def daily_locked_reason(self) -> Optional[str]:
        return self._daily_locked_reason

    @property
    def daily_net_rub(self) -> Decimal:
        return self._daily_net_rub

    @property
    def daily_hard_stops(self) -> int:
        return self._daily_hard_stops

    def _current_round_trip_fee(self, quantity: int = 1) -> Decimal:
        """Комиссия круга на текущих ценах (база для дневного лимита убытка)."""
        metrics = self._last_metrics
        if metrics is None:
            return ZERO
        return self._fees.estimated_round_trip_fee(
            self._leg1,
            metrics.micro_leg1,
            quantity,
            self._leg2,
            metrics.micro_leg2,
            self._topology,
        )

    def _register_daily_outcome(self, reason: ExitReason, net: Decimal, closed_ms: int) -> None:
        """Обновляет дневную статистику пары и при необходимости ставит блокировку."""
        params = self._params
        self._roll_daily(closed_ms)
        self._daily_net_rub += net
        if reason is ExitReason.HARD_STOP:
            self._daily_hard_stops += 1
        if not params.use_daily_stop_breaker or self._daily_locked_reason is not None:
            self._save_daily_state()
            return
        if self._daily_hard_stops >= max(int(params.max_daily_hard_stops), 1):
            self._daily_locked_reason = f"HardStop {self._daily_hard_stops} шт. за день"
        else:
            limit = -dec(params.max_daily_loss_fee_multiple) * self._current_round_trip_fee(
                params.quantity_lots
            )
            if limit < ZERO and self._daily_net_rub <= limit:
                self._daily_locked_reason = (
                    f"дневной убыток {self._daily_net_rub} RUB <= {limit} RUB "
                    f"({params.max_daily_loss_fee_multiple} x комиссия круга)"
                )
        if self._daily_locked_reason is not None:
            self._log.info(
                "Дневной предохранитель пары сработал: %s (net за день %s RUB)",
                self._daily_locked_reason,
                self._daily_net_rub,
            )
        self._save_daily_state()

    async def _maybe_enter(self, metrics: SpreadMetrics) -> None:
        params = self._params
        if self._bootstrap:
            # v2.4.1: во время прогрева только наполняем статистику, торговля запрещена.
            self._update_z_peak(metrics.zscore)
            self._update_regime(metrics)
            return
        zscore = metrics.zscore
        long_signal = zscore >= params.z_entry and metrics.obi_leg1 > -params.obi_entry_guard
        short_signal = zscore <= -params.z_entry and metrics.obi_leg1 < params.obi_entry_guard
        confirmed = long_signal or short_signal
        if params.use_entry_persistence:
            # v2.5: одиночный «фантомный» выброс Z (рассинхрон стаканов в REST) не должен
            # ни армировать пик для хука, ни открывать позицию. Требуем, чтобы условие
            # держалось не меньше (N−1) статистических интервалов (минимум 3 с).
            need = max(int(params.entry_confirm_samples), 1)
            hold_ms = max(
                (need - 1) * max(float(params.stat_sample_interval_seconds), 0.0), 3.0
            ) * 1000.0
            now_ms = int(metrics.time_ms)
            if not confirmed:
                self._entry_signal_since_ms = None
                self._update_z_peak(zscore, allow_arm=False)
                self._update_regime(metrics)
                return
            if self._entry_signal_since_ms is None:
                self._entry_signal_since_ms = now_ms
            armed = (now_ms - self._entry_signal_since_ms) >= hold_ms
            self._update_z_peak(zscore, allow_arm=armed)
            self._update_regime(metrics)
            if not armed:
                self._counters["persistenceSkips"] += 1
                self._log_throttled(
                    "persistence",
                    logging.DEBUG,
                    "Сигнал |Z|=%.2f не подтверждён: держится %.1f с из %.1f с",
                    abs(zscore),
                    (now_ms - self._entry_signal_since_ms) / 1000.0,
                    hold_ms / 1000.0,
                )
                return
        else:
            self._update_z_peak(zscore)
            self._update_regime(metrics)
        if not confirmed:
            return
        self._roll_daily(self._clock.now_ms())
        if self._daily_locked_reason is not None:
            # v2.4: после HardStop (или дневного лимита) пара не торгует до следующего дня.
            self._counters["dailyStopSkips"] += 1
            self._log_throttled(
                "daily-stop",
                logging.INFO,
                "Дневной предохранитель: %s — входы по паре заблокированы до конца дня",
                self._daily_locked_reason,
            )
            return
        if params.use_macro_beta_drift_guard:
            drift = metrics.beta_drift
            if drift is not None:
                if long_signal and drift > params.max_beta_drift:
                    # Leg1 структурно слабеет относительно Leg2 — не покупаем «падающий нож».
                    self._counters["betaDriftSkips"] += 1
                    self._log_throttled(
                        "beta-drift",
                        logging.DEBUG,
                        "Макро-дрейф беты %+.3f%% > %.3f%%: Long Leg1 заблокирован",
                        drift * 100.0,
                        params.max_beta_drift * 100.0,
                    )
                    return
                if short_signal and drift < -params.max_beta_drift:
                    self._counters["betaDriftSkips"] += 1
                    self._log_throttled(
                        "beta-drift",
                        logging.DEBUG,
                        "Макро-дрейф беты %+.3f%% < −%.3f%%: Short Leg1 заблокирован",
                        drift * 100.0,
                        params.max_beta_drift * 100.0,
                    )
                    return
        if params.use_z_peak_hook and not self._z_peak_confirmed(zscore):
            # v2.2: не входим на «падающий нож» — ждём разворота от пика |Z|.
            self._counters["hookSkips"] += 1
            self._log_throttled(
                "zpeak",
                logging.DEBUG,
                "Хук пика Z: |Z|=%.2f, пик=%.2f, откат=%.3f (нужно %.2f)",
                abs(zscore),
                self._z_peak,
                self._z_peak - abs(zscore),
                params.z_hook_pullback,
            )
            return
        if not self._regime_confirmed(zscore):
            # v2.2: ноги движутся разнонаправленным трендом — возврата к среднему не ждём.
            self._counters["regimeSkips"] += 1
            self._log_throttled(
                "regime",
                logging.DEBUG,
                "Режимный фильтр: расхождение тренда ног, вход заблокирован (Z=%.2f)",
                zscore,
            )
            return
        entry_cap = params.z_stop - max(params.entry_z_cap_slack, 0.0)
        if params.use_entry_z_cap and abs(zscore) > entry_cap:
            # Вход «в уже перегретый спред» приводил к мгновенному hard-stop по тому же Z.
            self._counters["overextendedSkips"] += 1
            self._log_throttled(
                "overextended",
                logging.DEBUG,
                "Пропуск: |Z|=%.2f выше предела входа %.2f (Z_stop=%.2f)",
                zscore,
                entry_cap,
                params.z_stop,
            )
            return
        self._counters["signals"] += 1
        side = Side.BUY if long_signal else Side.SELL
        quantity = self._guard.clamp_quantity(params.quantity_lots)
        entry_price = metrics.best_bid_leg1 if side is Side.BUY else metrics.best_ask_leg1
        if entry_price <= 0:
            return

        expected_gross = TinkoffFeeCalculator.expected_gross_reversion_rub(
            metrics.spread_rub, metrics.mean_rub, zscore, params.z_exit
        )
        if self._topology is ExecutionTopology.SINGLE_LEG_LEAD_LAG and metrics.beta > 1e-9:
            # v2.6.1: ожидаемый возврат спреда приводим к рублям Leg1 (делим на бету),
            # иначе фильтр покрытия комиссии сравнивает разные единицы.
            expected_gross = dec(expected_gross / dec(metrics.beta))
        # При лонге Leg1 хедж продаёт Leg2 (по биду), при шорте — покупает (по аску).
        leg2_reference = metrics.best_bid_leg2 if side is Side.BUY else metrics.best_ask_leg2
        fee_round_trip = self._fees.estimated_round_trip_fee(
            self._leg1,
            entry_price,
            quantity,
            self._leg2,
            leg2_reference if leg2_reference > 0 else metrics.micro_leg2,
            self._topology,
        )
        slippage = TinkoffFeeCalculator.slippage_buffer_rub(self._leg1, params.slippage_buffer_steps, quantity)
        if self._topology is ExecutionTopology.DUAL_LEG_HEDGED:
            slippage += TinkoffFeeCalculator.slippage_buffer_rub(self._leg2, params.slippage_buffer_steps, quantity)
        if params.use_vol_to_fee_filter and fee_round_trip > ZERO:
            sigma_leg1 = self._sigma_leg1_rub(metrics)
            ratio = float(sigma_leg1 / fee_round_trip)
            if ratio < params.min_sigma_to_fee_ratio:
                # v2.4: сжатие волатильности относительно рублёвой комиссии — стоим в стороне.
                self._counters["volFilterSkips"] += 1
                self._log_throttled(
                    "vol-fee",
                    logging.DEBUG,
                    "Фильтр σ/Fee: σ_leg1=%.4f RUB / fee=%s RUB = %.2f < %.2f",
                    float(sigma_leg1),
                    fee_round_trip,
                    ratio,
                    params.min_sigma_to_fee_ratio,
                )
                return
        if not TinkoffFeeCalculator.is_edge_sufficient(
            expected_gross, fee_round_trip, slippage, params.min_edge_multiplier
        ):
            self._counters["edgeSkips"] += 1
            self._log_throttled(
                "edge",
                logging.DEBUG,
                "Пропуск сигнала: ожидаемый возврат %s RUB < %s x (комиссия %s + слиппедж %s)",
                expected_gross,
                params.min_edge_multiplier,
                fee_round_trip,
                slippage,
            )
            return

        margin_legs = [self._leg1, self._leg2] if self._topology is ExecutionTopology.DUAL_LEG_HEDGED else [self._leg1]
        required_margin = self._guard.required_margin_rub(margin_legs, quantity)
        decision = self._guard.evaluate(self._clock.now_ms(), self._gateway.free_cash_rub(), required_margin)
        if not decision.allowed:
            self._counters["guardSkips"] += 1
            self._log_throttled("guard", logging.INFO, "Вход заблокирован риск-контролем: %s", decision.reason)
            return
        session_block = self._guard.entry_window_reason(self._clock.now_ms())
        if session_block is not None:
            self._counters["sessionSkips"] += 1
            self._log_throttled("session", logging.DEBUG, "Вход вне активного окна: %s", session_block)
            return

        self._trade_seq += 1
        position = OpenPosition(
            trade_id=self._trade_seq,
            opened_ms=self._clock.now_ms(),
            side_leg1=side,
            quantity=quantity,
            leg1_entry_price=entry_price,
            entry_z=zscore,
            margin_rub=required_margin,
        )
        self._position = position
        self._exit_reason = ExitReason.TAKE_PROFIT
        ack = await self._gateway.place_order(
            OrderRequest(
                leg=1,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=entry_price,
                client_tag=f"entry-{position.trade_id}",
            )
        )
        if not ack.accepted:
            self._position = None
            self._counters["rejects"] += 1
            self._log_throttled("entry-reject", logging.WARNING, "Не удалось выставить вход: %s", ack.message)
            return
        self._register_order(ROLE_ENTRY_LEG1, ack.order_id)
        self._last_requote_ms = self._clock.now_ms()
        self._counters["entries"] += 1
        self._transition(
            FsmState.QUOTING_ENTRY_LEG1,
            f"{side.value} Leg1 {quantity} @ {entry_price} (Z={zscore:.2f}, OBI={metrics.obi_leg1:.2f})",
        )

    async def _manage_entry_quote(self, metrics: SpreadMetrics) -> None:
        position = self._position
        order_id = self._roles.get(ROLE_ENTRY_LEG1)
        if position is None or order_id is None:
            return
        order = self._gateway.order(order_id)
        if order is None or not order.is_active:
            return
        if self._cancel_requested_role == ROLE_ENTRY_LEG1:
            return
        params = self._params
        zscore = metrics.zscore
        obi = metrics.obi_leg1
        side = position.side_leg1
        toxic = (side is Side.BUY and obi <= -params.obi_toxic_guard) or (
            side is Side.SELL and obi >= params.obi_toxic_guard
        )
        if abs(zscore) < params.z_cancel or toxic:
            self._cancel_requested_role = ROLE_ENTRY_LEG1
            await self._gateway.cancel_order(order_id)
            self._log.info(
                "Снятие входной котировки: Z=%.2f (cancel<%.2f), OBI=%.2f%s",
                zscore,
                params.z_cancel,
                obi,
                " — токсичный поток" if toxic else "",
            )
            return

        desired = metrics.best_bid_leg1 if side is Side.BUY else metrics.best_ask_leg1
        if desired <= 0 or order.price is None:
            return
        steps = abs(self._leg1.price_steps(order.price, desired))
        if steps < params.requote_step_threshold:
            return
        better = desired > order.price if side is Side.BUY else desired < order.price
        if not better:
            return
        now = self._clock.now_ms()
        if now - self._last_requote_ms < params.requote_cooldown_ms:
            return
        if abs(zscore) < params.z_entry - params.requote_z_slack:
            return
        ack = await self._gateway.replace_order(order_id, desired)
        if ack.accepted:
            self._last_requote_ms = now
            self._log.debug("Реквота входа: %s -> %s", order.price, desired)

    async def _manage_position(self, metrics: SpreadMetrics) -> None:
        position = self._position
        if position is None or position.tags.get("exitMarketSent"):
            return
        params = self._params
        now = self._clock.now_ms()
        hold_seconds = (now - position.opened_ms) / 1000.0
        z_live = metrics.zscore
        z_frozen = self._z_for_decision(position, metrics)
        unrealized = self._unrealized_gross(position, metrics)
        if unrealized > position.max_favorable_gross_rub:
            position.max_favorable_gross_rub = unrealized
        if unrealized < position.min_adverse_gross_rub:
            position.min_adverse_gross_rub = unrealized

        stop_reason = self._stop_reason(position, metrics, z_frozen, hold_seconds)
        if stop_reason is not None:
            await self._emergency_flatten(ExitReason.HARD_STOP, stop_reason)
            return
        if hold_seconds >= params.max_hold_seconds:
            if stop_reason is not None:
                await self._emergency_flatten(
                    ExitReason.HARD_STOP,
                    f"удержание {hold_seconds:.1f} c: {stop_reason}",
                )
            else:
                await self._start_timed_exit(ExitReason.TIME_STOP, metrics)
            return
        if await self._manage_breakeven_ratchet(position, metrics, z_frozen, unrealized):
            return
        await self._maybe_decay_tp(position)
        if self._topology is ExecutionTopology.SINGLE_LEG_LEAD_LAG and params.use_profit_lock:
            if await self._manage_ghost_reversion(position, metrics, z_live, unrealized):
                return
        if abs(z_frozen) <= params.z_exit and self._profit_floor_ok(position, unrealized):
            await self._start_passive_exit(ExitReason.TAKE_PROFIT, metrics)

    # ------------------------------------------------------------------ v2: экономические исправления
    def _z_for_decision(self, position: OpenPosition, metrics: SpreadMetrics) -> float:
        """Z-score для TP/SL: по замороженным на входе mu/sigma (v2) или живой (v1)."""
        if not self._params.use_frozen_mu or position.frozen_sigma <= 1e-9:
            return metrics.zscore
        return (metrics.spread_rub - position.frozen_mu) / position.frozen_sigma

    def _profit_floor_ok(self, position: OpenPosition, unrealized_gross_rub: Decimal) -> bool:
        """Правило 1: Z-выход разрешён только при покрытии издержек с запасом."""
        if not self._params.use_profit_lock:
            return True
        if position.round_trip_fee_rub <= 0:
            return True
        multiplier = (
            self._params.dual_profit_floor_multiplier
            if self._topology is ExecutionTopology.DUAL_LEG_HEDGED
            else self._params.profit_floor_multiplier
        )
        return unrealized_gross_rub >= position.round_trip_fee_rub * multiplier

    def _unrealized_gross(self, position: OpenPosition, metrics: SpreadMetrics) -> Decimal:
        """Текущий валовой PnL позиции по микропрайсам (в рублях)."""
        gross = ZERO
        long_leg1 = position.side_leg1 is Side.BUY
        qty1 = position.leg1_filled_qty - position.leg1_closed_qty
        if qty1 > 0:
            gross += (
                (metrics.micro_leg1 - position.leg1_entry_price)
                * self._leg1.rub_per_price_unit
                * Decimal(qty1)
                * Decimal(position.side_leg1.sign)
            )
        if self._topology is ExecutionTopology.DUAL_LEG_HEDGED and position.leg2_entry_price is not None:
            qty2 = position.leg2_filled_qty - position.leg2_closed_qty
            if qty2 > 0:
                gross += (
                    (metrics.micro_leg2 - position.leg2_entry_price)
                    * self._leg2.rub_per_price_unit
                    * Decimal(qty2)
                    * Decimal(-position.side_leg1.sign)
                )
        return gross

    def _leg1_unrealized_gross(self, position: OpenPosition, metrics: SpreadMetrics) -> Decimal:
        """Валовой PnL только по Leg1 (в рублях) — основа ценового стопа v2.3."""
        qty1 = position.leg1_filled_qty - position.leg1_closed_qty
        if qty1 <= 0:
            return ZERO
        return (
            (metrics.micro_leg1 - position.leg1_entry_price)
            * self._leg1.rub_per_price_unit
            * Decimal(qty1)
            * Decimal(position.side_leg1.sign)
        )

    def _single_leg_stop_reason(
        self,
        position: OpenPosition,
        metrics: SpreadMetrics,
        z_frozen: float,
        hold_seconds: float,
    ) -> Optional[str]:
        """v2.3: стоп для ``SingleLegLeadLag`` по фактическому убытку Leg1.

        Оценка по ``|z_frozen|`` для одиночной ноги некорректна: если Leg2 продолжает
        движение в нашу сторону, а Leg1 стоит, ``|Z|`` растёт, хотя убытка нет.
        Поэтому стоп срабатывает только когда:
        * просадка Leg1 превысила ``target × single_leg_stop_to_target_ratio``
          (не меньше ``single_leg_stop_min_steps`` шагов цены), либо
        * импульс ведущей ноги развернулся (``z_frozen`` ушёл за ``lead_failure_z``
          против позиции) при убытке Leg1 и удержании не меньше ``lead_failure_seconds``.
        """
        params = self._params
        leg1_gross = self._leg1_unrealized_gross(position, metrics)
        target_rub = position.tp_target_rub
        if target_rub <= 0:
            target_rub = self._target_tp_rub(position, self._clock.now_ms())
        step_rub = self._step_value_rub(position)
        ratio = Decimal(str(params.single_leg_stop_to_target_ratio))
        stop_rub = max(target_rub * ratio, step_rub * Decimal(max(params.single_leg_stop_min_steps, 1)))
        if leg1_gross <= -stop_rub:
            return (
                f"MAE Leg1 {leg1_gross} RUB <= -{stop_rub} RUB "
                f"(цель {target_rub} RUB × {params.single_leg_stop_to_target_ratio})"
            )
        adverse_z = z_frozen * position.side_leg1.sign
        if (
            leg1_gross < ZERO
            and hold_seconds >= params.lead_failure_seconds
            and adverse_z <= -params.lead_failure_z
        ):
            return (
                f"импульс Leg2 развернулся (Z_frozen={z_frozen:.2f}), Leg1 {leg1_gross} RUB, "
                f"{hold_seconds:.0f} c"
            )
        return None

    def _stop_reason(
        self,
        position: OpenPosition,
        metrics: SpreadMetrics,
        z_frozen: float,
        hold_seconds: float,
    ) -> Optional[str]:
        """Единая точка проверки стопа: по топологии и режиму v2.3."""
        params = self._params
        single_leg = self._topology is ExecutionTopology.SINGLE_LEG_LEAD_LAG
        if single_leg and params.use_single_leg_price_stop:
            return self._single_leg_stop_reason(position, metrics, z_frozen, hold_seconds)
        if abs(z_frozen) >= params.z_stop:
            return f"|Z_frozen|={abs(z_frozen):.2f} >= Z_stop={params.z_stop:.2f}"
        return None

    def _freeze_position(self, position: OpenPosition, metrics: Optional[SpreadMetrics]) -> None:
        """Фиксирует mu/sigma спреда на входе и оценивает комиссию круга."""
        if metrics is None:
            return
        position.frozen_mu = metrics.mean_rub
        position.frozen_sigma = math.sqrt(max(metrics.variance, 0.0))
        # v2.6.1: бета замораживается на входе — она нужна, чтобы переводить
        # девиацию спреда (в «спред-рублях») в рубли Leg1.
        position.frozen_beta = float(metrics.beta) if metrics.beta > 1e-9 else 1.0
        position.entry_spread = metrics.spread_rub
        position.leg1_flat_since_ms = self._clock.now_ms()
        leg2_reference = metrics.best_bid_leg2 if position.side_leg1 is Side.BUY else metrics.best_ask_leg2
        position.round_trip_fee_rub = self._fees.estimated_round_trip_fee(
            self._leg1,
            position.leg1_entry_price,
            max(position.leg1_filled_qty, 1),
            self._leg2,
            leg2_reference if leg2_reference > 0 else metrics.micro_leg2,
            self._topology,
        )
        self._log.info(
            "Зафиксированы параметры входа: mu=%s RUB, sigma=%.3f RUB, fee_rt=%s RUB",
            position.frozen_mu,
            position.frozen_sigma,
            position.round_trip_fee_rub,
        )

    async def _place_resting_tp(self, position: OpenPosition) -> None:
        """Правило 2: пассивная мейкерская заявка на прибыльном уровне в Leg1."""
        if not self._params.use_profit_lock:
            return
        remaining = position.leg1_filled_qty - position.leg1_closed_qty
        if remaining <= 0:
            return
        target_rub = self._target_tp_rub(position, self._clock.now_ms())
        price = self._tp_price_for(position, target_rub)
        if price is None:
            return
        ack = await self._gateway.place_order(
            OrderRequest(
                leg=1,
                side=position.side_leg1.opposite(),
                order_type=OrderType.LIMIT,
                quantity=remaining,
                price=price,
                client_tag=f"tp-{position.trade_id}",
            )
        )
        if not ack.accepted:
            self._log.warning("Не удалось выставить прибыльный тейк-профит: %s", ack.message)
            return
        self._register_order(ROLE_TP_LEG1, ack.order_id)
        position.tp_price = price
        position.tp_target_rub = target_rub
        position.tags["tpTargetRub"] = str(target_rub)
        self._log.info(
            "Мейкерский тейк-профит Leg1 @ %s (цель %s RUB, %.0f шагов)",
            price,
            target_rub,
            float(abs(price - position.leg1_entry_price) / self._leg1.min_price_increment),
        )

    def _target_tp_rub(self, position: OpenPosition, now_ms: int) -> Decimal:
        """Цель тейка в рублях Leg1 (v2.3: положительная выпуклость).

        ``target = max(tpMinFeeMultiple·Fee, tpDeviationFraction·|S_entry − mu| / β)`` —
        цель привязана к реальной девиации спреда, приведённой к рублям Leg1.
        Деление на бету обязательно (v2.6.1): ``S_t = V2 − β·V1`` измеряется в
        «спред-рублях», и без пересчёта цель уезжала в β раз дальше по цене, чем
        тот ход Leg1, который действительно закрывает спред (для VTBR/SBER β ≈ 5.1,
        для пар с крупным лотом — ещё сильнее).
        """
        params = self._params
        fee = position.round_trip_fee_rub
        beta = position.frozen_beta if position.frozen_beta > 1e-9 else 1.0
        deviation_rub = dec(
            abs(position.entry_spread - position.frozen_mu) * params.tp_deviation_fraction / beta
        )
        fee_floor = fee * params.tp_min_fee_multiple
        profit_floor = fee * params.min_net_profit_multiplier
        target = max(fee_floor, profit_floor, deviation_rub)
        if target <= 0:
            target = fee
        if not params.use_tp_decay or fee <= 0 or params.max_hold_seconds <= 0:
            return target
        decay_start = params.tp_decay_start_fraction * params.max_hold_seconds
        hold_seconds = max(now_ms - position.opened_ms, 0) / 1000.0
        if hold_seconds <= decay_start:
            return target
        span = max(params.max_hold_seconds - decay_start, 1.0)
        fraction = min((hold_seconds - decay_start) / span, 1.0)
        floor_rub = fee * params.tp_decay_floor_multiplier
        destination = floor_rub if floor_rub < target else target
        return target + (destination - target) * dec(fraction)

    def _tp_price_for(self, position: OpenPosition, target_rub: Decimal) -> Optional[Decimal]:
        """Переводит рублёвую цель в цену лимита с округлением вверх по шагу цены."""
        remaining = position.leg1_filled_qty - position.leg1_closed_qty
        if remaining <= 0:
            return None
        per_unit = self._leg1.rub_per_price_unit * Decimal(remaining)
        if per_unit <= 0:
            return None
        increment = self._leg1.min_price_increment
        delta = ceil_to_increment(target_rub / per_unit, increment)
        if delta <= 0:
            delta = increment
        price = round_to_increment(
            position.leg1_entry_price + Decimal(position.side_leg1.sign) * delta,
            increment,
        )
        if position.side_leg1 is Side.BUY and price <= position.leg1_entry_price:
            price = position.leg1_entry_price + increment
        if position.side_leg1 is Side.SELL and price >= position.leg1_entry_price:
            price = position.leg1_entry_price - increment
        return price

    async def _maybe_decay_tp(self, position: OpenPosition) -> None:
        """v2.2: сжимает цель тейка по времени, подтягивая мейкерский лимит к рынку."""
        params = self._params
        if not params.use_tp_decay or not params.use_profit_lock:
            return
        order_id = self._roles.get(ROLE_TP_LEG1)
        if order_id is None:
            return
        order = self._gateway.order(order_id)
        if order is None or not order.is_active or order.price is None:
            return
        target_rub = self._target_tp_rub(position, self._clock.now_ms())
        price = self._tp_price_for(position, target_rub)
        if price is None or price == order.price:
            return
        better = price < order.price if position.side_leg1 is Side.BUY else price > order.price
        if not better:
            return
        ack = await self._gateway.replace_order(order_id, price)
        if not ack.accepted:
            return
        position.tp_price = price
        position.tp_target_rub = target_rub
        self._log.debug(
            "Сжатие тейка: %s -> %s (цель %s RUB, удержание %.0f c)",
            order.price,
            price,
            target_rub,
            (self._clock.now_ms() - position.opened_ms) / 1000.0,
        )

    async def _manage_breakeven_ratchet(
        self,
        position: OpenPosition,
        metrics: SpreadMetrics,
        z_frozen: float,
        unrealized: Decimal,
    ) -> bool:
        """v2.3: фиксирует прибыль только при провале от глубокого пика.

        Взвод — `max(2.2·Fee, 0.90·TargetGross) + 1 шаг`; выход — откат не меньше
        2 шагов, прибыль упала до `max(1.15·Fee, 0.60·TargetGross)` и позиция
        удерживается не меньше 120 с. По умолчанию ratchet выключен: он срезает
        выпуклость (см. отчёт v2.3, §2).
        """
        params = self._params
        if not params.use_breakeven_ratchet or not params.use_profit_lock:
            return False
        if self._topology is not ExecutionTopology.SINGLE_LEG_LEAD_LAG:
            return False
        fee = position.round_trip_fee_rub
        if fee <= 0:
            return False
        step_rub = self._step_value_rub(position)
        if not position.breakeven_locked:
            target_rub = position.tp_target_rub or self._target_tp_rub(position, self._clock.now_ms())
            arm_rub = max(fee * params.ratchet_arm_multiplier, target_rub * dec(params.ratchet_arm_target_fraction))
            arm_rub += step_rub * params.ratchet_arm_step_buffer
            if position.max_favorable_gross_rub >= arm_rub:
                position.breakeven_locked = True
                position.tags["ratchetArmedRub"] = str(position.max_favorable_gross_rub)
                position.tags["ratchetArmThresholdRub"] = str(arm_rub)
                self._counters["ratchetExits"] += 1
                self._log.info(
                    "Breakeven-ratchet взведён: MFE=%s RUB >= порог %s RUB (fee_rt=%s RUB, цель=%s RUB)",
                    position.max_favorable_gross_rub,
                    arm_rub,
                    fee,
                    target_rub,
                )
            return False
        hold_seconds = (self._clock.now_ms() - position.opened_ms) / 1000.0
        if hold_seconds < params.ratchet_min_hold_seconds:
            return False
        retrace_limit = max(
            fee * params.ratchet_exit_multiplier,
            (position.tp_target_rub or fee) * dec(params.ratchet_exit_target_fraction),
        )
        retrace_rub = position.max_favorable_gross_rub - unrealized
        min_retrace = step_rub * Decimal(max(params.ratchet_retrace_steps, 1))
        if unrealized <= retrace_limit and retrace_rub >= min_retrace:
            self._log.info(
                "Breakeven-ratchet: фиксируем прибыль (MFE=%s -> %s RUB, откат=%s RUB >= %s RUB, "
                "лимит=%s RUB, удержание %.0f c)",
                position.max_favorable_gross_rub,
                unrealized,
                retrace_rub,
                min_retrace,
                retrace_limit,
                hold_seconds,
            )
            await self._start_passive_exit(ExitReason.BREAKEVEN, metrics)
            return True
        return False

    def _step_value_rub(self, position: OpenPosition) -> Decimal:
        """Стоимость одного шага цены Leg1 в рублях на текущий объём позиции."""
        quantity = position.leg1_filled_qty - position.leg1_closed_qty
        if quantity <= 0:
            quantity = max(position.leg1_filled_qty, 1)
        return self._leg1.rub_per_price_unit * self._leg1.min_price_increment * Decimal(quantity)

    async def _manage_ghost_reversion(
        self,
        position: OpenPosition,
        metrics: SpreadMetrics,
        z_live: float,
        unrealized: Decimal,
    ) -> bool:
        """Правило 3: не «выбрасывать» Leg1 рынком, если реверсия пришла только от Leg2."""
        params = self._params
        now = self._clock.now_ms()
        increment = self._leg1.min_price_increment
        if abs(metrics.micro_leg1 - position.leg1_entry_price) >= increment:
            position.leg1_flat_since_ms = now

        if position.tags.get("scratch"):
            if now >= position.scratch_until_ms:
                self._exit_reason = ExitReason.TIME_STOP
                await self._market_exit_leg1("скретч-выход не исполнился за отведённое время")
            return True

        flat_for_ms = now - position.leg1_flat_since_ms
        if (
            abs(z_live) < params.ghost_reversion_z
            and flat_for_ms >= params.ghost_flat_seconds * 1000
            and unrealized < position.round_trip_fee_rub
        ):
            await self._place_scratch_exit(position)
            return True
        return False

    async def _place_scratch_exit(self, position: OpenPosition) -> None:
        """Скретч-выход: пассивный лимит на entry ± 1 шаг вместо рыночного дампа."""
        tp_order = self._roles.get(ROLE_TP_LEG1)
        if tp_order is not None:
            await self._gateway.cancel_order(tp_order)
        self._release_role(ROLE_TP_LEG1)
        remaining = position.leg1_filled_qty - position.leg1_closed_qty
        if remaining <= 0:
            return
        price = shift_by_steps(
            position.leg1_entry_price,
            position.side_leg1.sign,
            self._leg1.min_price_increment,
        )
        ack = await self._gateway.place_order(
            OrderRequest(
                leg=1,
                side=position.side_leg1.opposite(),
                order_type=OrderType.LIMIT,
                quantity=remaining,
                price=price,
                client_tag=f"scratch-{position.trade_id}",
            )
        )
        if not ack.accepted:
            self._log.warning("Скретч-заявка отклонена: %s", ack.message)
            return
        self._register_order(ROLE_TP_LEG1, ack.order_id)
        position.tp_price = price
        position.scratch_until_ms = self._clock.now_ms() + self._params.scratch_seconds * 1000
        position.tags["scratch"] = True
        self._exit_reason = ExitReason.SCRATCH
        self._log.info(
            "Ghost-reversion: Leg2 вернулась, Leg1 стоит — скретч-лимит @ %s на %d c",
            price,
            self._params.scratch_seconds,
        )

    async def _manage_exit_quote(self, metrics: SpreadMetrics) -> None:
        position = self._position
        order_id = self._roles.get(ROLE_EXIT_LEG1)
        if position is None or order_id is None:
            return
        if position.tags.get("exitMarketSent"):
            return
        order = self._gateway.order(order_id)
        if order is None or not order.is_active:
            return
        params = self._params
        z_frozen = self._z_for_decision(position, metrics)
        if abs(z_frozen) >= params.z_stop:
            await self._emergency_flatten(
                ExitReason.HARD_STOP,
                f"Z снова ушёл против позиции: {z_frozen:.2f}",
            )
            return
        side = position.side_leg1.opposite()
        desired = metrics.best_bid_leg1 if side is Side.BUY else metrics.best_ask_leg1
        if desired > 0 and order.price is not None:
            steps = abs(self._leg1.price_steps(order.price, desired))
            better = desired < order.price if side is Side.BUY else desired > order.price
            now = self._clock.now_ms()
            if (
                better
                and steps >= params.requote_step_threshold
                and now - self._last_exit_requote_ms >= params.requote_cooldown_ms
            ):
                ack = await self._gateway.replace_order(order_id, desired)
                if ack.accepted:
                    self._last_exit_requote_ms = now
        if self._clock.now_ms() >= position.exit_deadline_ms:
            await self._market_exit_leg1("таймаут пассивного выхода")

    # ------------------------------------------------------------------ исполнение
    async def _handle_fill(self, event: ExecutionEvent) -> None:
        role = self._role_by_order.get(event.order_id)
        position = self._position
        if role is None or position is None:
            return
        if role == ROLE_ENTRY_LEG1:
            await self._on_entry_fill(position, event)
        elif role == ROLE_HEDGE_LEG2:
            await self._on_hedge_fill(position, event)
        elif role in _LEG1_CLOSE_ROLES:
            await self._on_leg1_exit_fill(position, event)
        elif role in _LEG2_CLOSE_ROLES:
            await self._on_leg2_exit_fill(position, event)

    async def _on_entry_fill(self, position: OpenPosition, event: ExecutionEvent) -> None:
        if self._state is not FsmState.QUOTING_ENTRY_LEG1:
            return
        previous = position.leg1_filled_qty
        position.leg1_filled_qty += event.quantity
        position.leg1_entry_price = _blend_price(position.leg1_entry_price, previous, event.price, event.quantity)
        position.fees_rub += event.commission_rub
        self._log.info(
            "Исполнение входа Leg1: %d/%d @ %s (%s)",
            position.leg1_filled_qty,
            position.quantity,
            event.price,
            event.liquidity.value,
        )
        order = self._gateway.order(event.order_id)
        if (order is not None and order.is_active) or event.remaining > 0:
            self._cancel_requested_role = ROLE_ENTRY_LEG1
            await self._gateway.cancel_order(event.order_id)
        self._release_role(ROLE_ENTRY_LEG1)
        if self._topology is ExecutionTopology.DUAL_LEG_HEDGED:
            await self._hedge_entry("хедж после входа")
        else:
            position.margin_rub = self._leg1.position_margin_rub(position.leg1_filled_qty)
            self._freeze_position(position, self._last_metrics)
            await self._place_resting_tp(position)
            self._transition(
                FsmState.IN_POSITION,
                f"позиция открыта: {position.side_leg1.value} {position.leg1_filled_qty} @ {position.leg1_entry_price}",
            )

    async def _hedge_entry(self, reason: str) -> None:
        position = self._position
        if position is None:
            return
        need = position.leg1_filled_qty - position.leg2_filled_qty
        if need <= 0:
            self._release_role(ROLE_HEDGE_LEG2)
            self._transition(FsmState.IN_POSITION, reason)
            return
        previous = self._roles.get(ROLE_HEDGE_LEG2)
        if previous is not None:
            await self._gateway.cancel_order(previous)
        self._release_role(ROLE_HEDGE_LEG2)
        params = self._params
        side = position.side_leg1.opposite()
        attempt = int(position.tags.get("hedgeAttempts", 0)) + 1
        position.tags["hedgeAttempts"] = attempt
        metrics = self._last_metrics
        order_type = OrderType.MARKET
        price: Optional[Decimal] = None
        if attempt == 1 and metrics is not None:
            reference = metrics.best_ask_leg2 if side is Side.BUY else metrics.best_bid_leg2
            if reference > 0:
                steps = params.slippage_buffer_steps if side is Side.BUY else -params.slippage_buffer_steps
                price = shift_by_steps(reference, steps, self._leg2.min_price_increment)
                order_type = OrderType.LIMIT_IOC
        ack = await self._gateway.place_order(
            OrderRequest(
                leg=2,
                side=side,
                order_type=order_type,
                quantity=need,
                price=price,
                client_tag=f"hedge-{position.trade_id}",
            )
        )
        if not ack.accepted and order_type is OrderType.LIMIT_IOC:
            ack = await self._gateway.place_order(
                OrderRequest(2, side, OrderType.MARKET, need, None, f"hedge-mkt-{position.trade_id}")
            )
        if not ack.accepted:
            self._log.error("Хедж Leg2 отклонён: %s — аварийное закрытие", ack.message)
            await self._emergency_flatten(ExitReason.EMERGENCY, "хедж не выставлен")
            return
        self._register_order(ROLE_HEDGE_LEG2, ack.order_id)
        position.hedge_started_ms = self._clock.now_ms()
        self._transition(FsmState.HEDGING_ENTRY_LEG2, f"{reason}: {side.value} Leg2 {need}")

    async def _on_hedge_fill(self, position: OpenPosition, event: ExecutionEvent) -> None:
        previous = position.leg2_filled_qty
        position.leg2_filled_qty += event.quantity
        position.leg2_entry_price = _blend_price(position.leg2_entry_price, previous, event.price, event.quantity)
        position.fees_rub += event.commission_rub
        self._log.info(
            "Исполнение хеджа Leg2: %d/%d @ %s",
            position.leg2_filled_qty,
            position.leg1_filled_qty,
            event.price,
        )
        if position.leg2_filled_qty < position.leg1_filled_qty:
            await self._hedge_entry("добор хеджа")
            return
        self._release_role(ROLE_HEDGE_LEG2)
        position.margin_rub = self._leg1.position_margin_rub(position.leg1_filled_qty) + self._leg2.position_margin_rub(
            position.leg2_filled_qty
        )
        self._freeze_position(position, self._last_metrics)
        self._transition(
            FsmState.IN_POSITION,
            f"хедж закрыт: Leg1 {position.leg1_entry_price} / Leg2 {position.leg2_entry_price}",
        )

    async def _start_passive_exit(
        self,
        reason: ExitReason,
        metrics: SpreadMetrics,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        position = self._position
        if position is None:
            return
        tp_order = self._roles.get(ROLE_TP_LEG1)
        if tp_order is not None:
            await self._gateway.cancel_order(tp_order)
        self._release_role(ROLE_TP_LEG1)
        remaining = position.leg1_filled_qty - position.leg1_closed_qty
        if remaining <= 0:
            await self._after_leg1_closed()
            return
        side = position.side_leg1.opposite()
        price = metrics.best_bid_leg1 if side is Side.BUY else metrics.best_ask_leg1
        if price <= 0:
            await self._market_exit_leg1("нет встречной цены")
            return
        self._exit_reason = reason
        self._release_role(ROLE_EXIT_LEG1)
        self._release_role(ROLE_EXIT_LEG1_MARKET)
        ack = await self._gateway.place_order(
            OrderRequest(1, side, OrderType.LIMIT, remaining, price=price, client_tag=f"exit-{position.trade_id}")
        )
        if not ack.accepted:
            await self._market_exit_leg1(f"пассивный выход отклонён: {ack.message}")
            return
        self._register_order(ROLE_EXIT_LEG1, ack.order_id)
        now = self._clock.now_ms()
        position.exit_quote_started_ms = now
        deadline_seconds = timeout_seconds if timeout_seconds is not None else self._params.max_exit_quote_seconds
        position.exit_deadline_ms = now + max(int(deadline_seconds), 1) * 1000
        position.tags.pop("scratch", None)
        position.tags["quoteExit"] = True
        self._last_exit_requote_ms = now
        self._transition(
            FsmState.QUOTING_EXIT_LEG1,
            f"{reason.value}: {side.value} {remaining} @ {price} (Z={metrics.zscore:.2f}, "
            f"пассив до {deadline_seconds} c)",
        )

    async def _start_timed_exit(self, reason: ExitReason, metrics: Optional[SpreadMetrics]) -> None:
        """Мейкер-выход по тайм-стопу: 30 с пассивной котировки, затем рынок."""
        if metrics is None:
            self._exit_reason = reason
            await self._market_exit_leg1("нет метрик для пассивного выхода")
            return
        await self._start_passive_exit(reason, metrics, self._params.time_stop_quote_seconds)

    async def _market_exit_leg1(self, reason: str, force: bool = False) -> None:
        position = self._position
        if position is None:
            return
        if position.tags.get("exitMarketSent") and not force:
            return
        self._release_role(ROLE_EXIT_LEG1)
        remaining = position.leg1_filled_qty - position.leg1_closed_qty
        if remaining <= 0:
            await self._after_leg1_closed()
            return
        side = position.side_leg1.opposite()
        ack = await self._gateway.place_order(
            OrderRequest(1, side, OrderType.MARKET, remaining, None, f"exit-mkt-{position.trade_id}")
        )
        if not ack.accepted:
            self._log.error("Рыночный выход Leg1 отклонён: %s", ack.message)
            await self._emergency_flatten(ExitReason.EMERGENCY, "выход не выставлен")
            return
        position.tags["exitMarketSent"] = True
        self._register_order(ROLE_EXIT_LEG1_MARKET, ack.order_id)
        self._log.info("Рыночный выход Leg1: %s %d (%s)", side.value, remaining, reason)

    async def _on_leg1_exit_fill(self, position: OpenPosition, event: ExecutionEvent) -> None:
        role = self._role_by_order.get(event.order_id)
        previous = position.leg1_closed_qty
        position.leg1_closed_qty += event.quantity
        position.leg1_exit_price = _blend_price(position.leg1_exit_price, previous, event.price, event.quantity)
        position.fees_rub += event.commission_rub
        self._log.info(
            "Исполнение выхода Leg1: %d/%d @ %s (%s)",
            position.leg1_closed_qty,
            position.leg1_filled_qty,
            event.price,
            role or "?",
        )
        if position.leg1_closed_qty < position.leg1_filled_qty:
            if role == ROLE_TP_LEG1:
                # Пассивный тейк/скретч продолжает стоять до полного исполнения.
                return
            if role == ROLE_EXIT_LEG1:
                await self._gateway.cancel_order(event.order_id)
            self._release_role(ROLE_EXIT_LEG1)
            await self._market_exit_leg1("частичное исполнение выхода", force=True)
            return
        self._release_role(ROLE_EXIT_LEG1)
        self._release_role(ROLE_EXIT_LEG1_MARKET)
        self._release_role(ROLE_TP_LEG1)
        await self._after_leg1_closed()

    async def _after_leg1_closed(self) -> None:
        position = self._position
        if position is None:
            return
        need_leg2 = position.leg2_filled_qty - position.leg2_closed_qty
        if self._topology is ExecutionTopology.DUAL_LEG_HEDGED and need_leg2 > 0:
            await self._hedge_exit_leg2()
            return
        await self._finish_trade(self._exit_reason)

    async def _hedge_exit_leg2(self) -> None:
        position = self._position
        if position is None:
            return
        need = position.leg2_filled_qty - position.leg2_closed_qty
        if need <= 0:
            await self._finish_trade(self._exit_reason)
            return
        side = position.side_leg1
        ack = await self._gateway.place_order(
            OrderRequest(2, side, OrderType.MARKET, need, None, f"exit-leg2-{position.trade_id}")
        )
        if not ack.accepted:
            self._log.error("Закрытие Leg2 отклонено: %s", ack.message)
            await self._emergency_flatten(ExitReason.EMERGENCY, "закрытие хеджа не выставлено")
            return
        self._register_order(ROLE_EXIT_LEG2, ack.order_id)
        self._transition(FsmState.HEDGING_EXIT_LEG2, f"закрытие хеджа Leg2: {side.value} {need}")

    async def _on_leg2_exit_fill(self, position: OpenPosition, event: ExecutionEvent) -> None:
        previous = position.leg2_closed_qty
        position.leg2_closed_qty += event.quantity
        position.leg2_exit_price = _blend_price(position.leg2_exit_price, previous, event.price, event.quantity)
        position.fees_rub += event.commission_rub
        if position.leg2_closed_qty < position.leg2_filled_qty:
            await self._hedge_exit_leg2()
            return
        self._release_role(ROLE_EXIT_LEG2)
        await self._finish_trade(self._exit_reason)

    async def _emergency_flatten(self, reason: ExitReason, detail: str) -> None:
        position = self._position
        if position is None:
            return
        self._exit_reason = reason
        for role in list(self._roles.keys()):
            order_id = self._roles.get(role)
            if order_id is not None:
                await self._gateway.cancel_order(order_id)
            self._release_role(role)
        self._log.warning("Аварийное закрытие (%s): %s", reason.value, detail)

        remaining_leg1 = position.leg1_filled_qty - position.leg1_closed_qty
        remaining_leg2 = position.leg2_filled_qty - position.leg2_closed_qty
        if remaining_leg1 <= 0 and remaining_leg2 <= 0:
            await self._finish_trade(reason)
            return
        if remaining_leg1 > 0:
            ack = await self._gateway.place_order(
                OrderRequest(1, position.side_leg1.opposite(), OrderType.MARKET, remaining_leg1, None, f"flatten1-{position.trade_id}")
            )
            if ack.accepted:
                self._register_order(ROLE_FLATTEN_LEG1, ack.order_id)
            else:
                self._log.error("Не удалось закрыть Leg1 рынком: %s", ack.message)
        if remaining_leg2 > 0:
            ack = await self._gateway.place_order(
                OrderRequest(2, position.side_leg1, OrderType.MARKET, remaining_leg2, None, f"flatten2-{position.trade_id}")
            )
            if ack.accepted:
                self._register_order(ROLE_FLATTEN_LEG2, ack.order_id)
            else:
                self._log.error("Не удалось закрыть Leg2 рынком: %s", ack.message)
        self._transition(FsmState.EMERGENCY_FLATTEN, f"{reason.value}: {detail}")

    async def _finish_trade(self, reason: ExitReason) -> None:
        position = self._position
        if position is None:
            return
        closed_ms = self._clock.now_ms()
        gross = ZERO
        quantity_leg1 = position.leg1_closed_qty
        if quantity_leg1 > 0 and position.leg1_exit_price is not None:
            gross += (
                (position.leg1_exit_price - position.leg1_entry_price)
                * self._leg1.rub_per_price_unit
                * Decimal(quantity_leg1)
                * Decimal(position.side_leg1.sign)
            )
        quantity_leg2 = position.leg2_closed_qty
        if (
            self._topology is ExecutionTopology.DUAL_LEG_HEDGED
            and quantity_leg2 > 0
            and position.leg2_exit_price is not None
            and position.leg2_entry_price is not None
        ):
            gross += (
                (position.leg2_exit_price - position.leg2_entry_price)
                * self._leg2.rub_per_price_unit
                * Decimal(quantity_leg2)
                * Decimal(-position.side_leg1.sign)
            )
        fees = position.fees_rub
        net = gross - fees
        trade = TradeRecord(
            trade_id=position.trade_id,
            opened_ms=position.opened_ms,
            closed_ms=closed_ms,
            topology=self._topology,
            side_leg1=position.side_leg1,
            quantity=max(quantity_leg1, 0),
            leg1_entry_price=position.leg1_entry_price,
            leg1_exit_price=position.leg1_exit_price or position.leg1_entry_price,
            leg2_entry_price=position.leg2_entry_price,
            leg2_exit_price=position.leg2_exit_price,
            gross_pnl_rub=gross,
            fees_rub=fees,
            net_pnl_rub=net,
            entry_z=position.entry_z,
            exit_z=self._last_metrics.zscore if self._last_metrics else 0.0,
            margin_rub=position.margin_rub,
            exit_reason=reason,
            mfe_rub=position.max_favorable_gross_rub,
            mae_rub=position.min_adverse_gross_rub,
        )
        self._ledger.register(trade)
        self._guard.register_trade_pnl(net, closed_ms)
        self._register_daily_outcome(reason, net, closed_ms)
        self._counters["trades"] += 1
        self._log.info(
            "Сделка #%d закрыта: %s | %s %d | gross=%s RUB | fees=%s RUB | net=%s RUB | удержание %.1f c",
            trade.trade_id,
            reason.value,
            trade.side_leg1.value,
            trade.quantity,
            trade.gross_pnl_rub,
            trade.fees_rub,
            trade.net_pnl_rub,
            trade.hold_seconds,
        )
        self._position = None
        self._roles.clear()
        self._role_by_order.clear()
        self._cancel_requested_role = None
        self._calc.unfreeze_beta()
        self._transition(FsmState.COOLDOWN, f"{reason.value}: net {net} RUB, пауза {self._params.cooldown_seconds} c")

    # ------------------------------------------------------------------ таймеры
    async def _on_time(self) -> None:
        position = self._position
        now = self._clock.now_ms()
        params = self._params
        if self._state is FsmState.COOLDOWN:
            if now - self._state_since_ms >= params.cooldown_seconds * 1000:
                self._transition(FsmState.IDLE, "пауза завершена")
            return
        if position is None:
            return
        if self._state is FsmState.QUOTING_EXIT_LEG1:
            if now >= position.exit_deadline_ms:
                await self._market_exit_leg1("таймаут пассивного выхода")
            return
        if self._state is FsmState.IN_POSITION:
            if position.tags.get("scratch") and now >= position.scratch_until_ms:
                await self._start_timed_exit(ExitReason.SCRATCH, self._last_metrics)
                return
            if now - position.opened_ms >= params.max_hold_seconds * 1000:
                z_frozen = (
                    self._z_for_decision(position, self._last_metrics)
                    if self._last_metrics is not None
                    else 0.0
                )
                if abs(z_frozen) >= params.z_stop:
                    await self._emergency_flatten(
                        ExitReason.HARD_STOP, "истекло время удержания при |Z| >= Z_stop"
                    )
                else:
                    await self._start_timed_exit(ExitReason.TIME_STOP, self._last_metrics)
            return
        if self._state is FsmState.HEDGING_ENTRY_LEG2:
            if now - position.hedge_started_ms >= params.hedge_timeout_ms:
                await self._hedge_entry("таймаут хеджа")
            return
        if self._state is FsmState.EMERGENCY_FLATTEN:
            if not self._roles and now - self._state_since_ms >= params.flatten_retry_ms:
                await self._emergency_flatten(self._exit_reason, "повтор аварийного закрытия")
            return
