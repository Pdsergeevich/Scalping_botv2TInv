"""Конфигурация приложения: ``appsettings.json`` -> типизированные настройки."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from bot_a.economics.fees import Tariff
from bot_a.engine.fsm import StrategyParams
from bot_a.models.domain import ExecutionTopology, InstrumentMetadata
from bot_a.utils.quotation import dec

DEFAULT_CONFIG_NAME = "appsettings.json"


@dataclass(frozen=True, slots=True)
class TInvestSettings:
    token: str = ""
    token_env_var: str = "TINVEST_TOKEN"
    token_file: str = ""
    app_name: str = "BotA-Scalper"
    request_timeout_seconds: int = 20
    account_id: str = ""

    def resolve_token(self, base_dir: Optional[str | Path] = None) -> str:
        from bot_a.utils.secrets import read_token

        return read_token(
            explicit=self.token,
            env_var=self.token_env_var,
            base_dir=base_dir,
            file_hint=self.token_file or None,
        )

    def token_source(self, base_dir: Optional[str | Path] = None) -> str:
        from bot_a.utils.secrets import token_source

        return token_source(self.token, self.token_env_var, base_dir)


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    level: str = "Information"
    log_dir: str = "./logs"
    file_name: str = "bot_a.log"


@dataclass(frozen=True, slots=True)
class FeeSettings:
    tariff: Tariff = Tariff.TRADER


@dataclass(frozen=True, slots=True)
class RiskSettings:
    max_daily_drawdown_rub: Decimal = Decimal("15000")
    margin_safety_buffer_pct: float = 25.0
    max_position_lots: int = 2
    trade_evening_session: bool = True
    flatten_on_shutdown: bool = True
    active_session_windows: str = "AllDay"


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    topology: ExecutionTopology = ExecutionTopology.SINGLE_LEG_LEAD_LAG
    simulated_order_latency_ms: int = 25
    queue_fill_factor: float = 0.35
    slippage_steps: int = 1
    order_book_depth: int = 5
    paper_initial_cash_rub: Decimal = Decimal("300000")
    order_poll_interval_ms: int = 250
    cash_poll_interval_ms: int = 3000
    stat_sample_interval_seconds: float = 5.0
    bootstrap_warmup_days: int = 5
    # v2.5: защита от фантомных выбросов Z в REST/L2-режимах
    l2_entry_persistence: bool = True
    l2_entry_confirm_samples: int = 2
    l2_skew_factor: float = 1.5


@dataclass(frozen=True, slots=True)
class StorageSettings:
    postgres_dsn: str = ""
    recordings_dir: str = "./recordings"
    data_cache_dir: str = "./data_cache"
    reports_dir: str = "./reports"
    state_dir: str = "./state"
    record_to_postgres: bool = True
    record_to_files: bool = True
    recording_batch_size: int = 500
    recording_flush_seconds: float = 2.0


@dataclass(frozen=True, slots=True)
class PairPreset:
    name: str
    description: str
    leg1: InstrumentMetadata
    leg2: InstrumentMetadata
    beta: float = 1.0
    beta_mode: str = "same_underlying"
    enabled: bool = True
    z_entry: Optional[float] = None
    topology: Optional[str] = None
    active_session_windows: Optional[str] = None
    # v2.6: персональные настройки пары — окно EWMA и порог волатильность/комиссия
    ewma_window: Optional[int] = None
    min_sigma_to_fee_ratio: Optional[float] = None
    # v2.6.2: форма тейка и размер позиции
    tp_deviation_fraction: Optional[float] = None
    tp_min_fee_multiple: Optional[float] = None
    quantity_lots: Optional[int] = None

    def with_metadata(self, leg1: InstrumentMetadata, leg2: InstrumentMetadata) -> "PairPreset":
        return replace(self, leg1=leg1, leg2=leg2)


def _align_stat_cadence(params: StrategyParams, seconds: float) -> StrategyParams:
    """v2.5: единая каденция статистики во всех режимах.

    Раньше бэктест двигал EWMA на каждом событии (книга + принт), а live/replay —
    раз в 5 секунд. Из-за этого окно ``EwmaWindow`` в бэктесте покрывало вдвое
    меньше времени, и `Z` получался другим. Теперь значение берётся из
    ``execution.statSampleIntervalSeconds`` для любого режима.
    """
    if params.stat_sample_interval_seconds > 0 or seconds <= 0:
        return params
    return replace(params, stat_sample_interval_seconds=float(seconds))


@dataclass(frozen=True, slots=True)
class AppConfig:
    t_invest: TInvestSettings = field(default_factory=TInvestSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    fees: FeeSettings = field(default_factory=FeeSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    strategy: StrategyParams = field(default_factory=StrategyParams)
    presets: Dict[str, PairPreset] = field(default_factory=dict)
    active_preset: str = ""
    source_path: Optional[str] = None
    base_dir: str = "."

    # ------------------------------------------------------------------ доступ
    def preset(self, name: Optional[str] = None) -> PairPreset:
        key = name or self.active_preset
        if not key:
            if len(self.presets) == 1:
                return next(iter(self.presets.values()))
            raise KeyError("Не задан активный пресет пары (activePreset)")
        if key not in self.presets:
            available = ", ".join(sorted(self.presets.keys()))
            raise KeyError(f"Пресет '{key}' не найден. Доступны: {available}")
        return self.presets[key]

    def preset_names(self) -> list[str]:
        return sorted(self.presets.keys())

    def trading_preset_names(self) -> list[str]:
        """Пары, разрешённые к реальной торговле (``enabled = true``)."""
        return sorted(name for name, preset in self.presets.items() if preset.enabled)

    def disabled_preset_names(self) -> list[str]:
        return sorted(name for name, preset in self.presets.items() if not preset.enabled)

    def path(self, value: str) -> Path:
        """Разрешает относительные пути относительно каталога конфигурации."""
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return Path(self.base_dir) / candidate

    def ensure_directories(self) -> None:
        for value in (
            self.logging.log_dir,
            self.storage.recordings_dir,
            self.storage.data_cache_dir,
            self.storage.reports_dir,
            self.storage.state_dir,
        ):
            self.path(value).mkdir(parents=True, exist_ok=True)

    def with_overrides(
        self,
        tariff: Optional[Tariff] = None,
        topology: Optional[ExecutionTopology] = None,
        strategy: Optional[StrategyParams] = None,
        active_preset: Optional[str] = None,
        max_daily_drawdown_rub: Optional[Decimal] = None,
        token: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> "AppConfig":
        fees = replace(self.fees, tariff=tariff) if tariff is not None else self.fees
        execution = replace(self.execution, topology=topology) if topology is not None else self.execution
        risk = (
            replace(self.risk, max_daily_drawdown_rub=dec(max_daily_drawdown_rub))
            if max_daily_drawdown_rub is not None
            else self.risk
        )
        t_invest = self.t_invest
        if token is not None or account_id is not None:
            t_invest = replace(
                self.t_invest,
                token=token if token is not None else self.t_invest.token,
                account_id=account_id if account_id is not None else self.t_invest.account_id,
            )
        return replace(
            self,
            fees=fees,
            execution=execution,
            risk=risk,
            strategy=strategy or self.strategy,
            active_preset=active_preset or self.active_preset,
            t_invest=t_invest,
        )

    def with_preset_overrides(self, name: Optional[str] = None) -> "AppConfig":
        """Применяет per-preset настройки пары.

        Поддерживаются ``zEntry``, ``topology``, ``activeSessionWindows``, а с v2.6 —
        ещё ``ewmaWindow`` и ``minSigmaToFeeRatio``: у фьючерсных календарных спредов
        волатильность к комиссии ниже, чем у акций, и глобальный порог 1.30 их душит.
        """
        preset = self.preset(name)
        strategy = self.strategy
        if preset.z_entry is not None:
            strategy = strategy.with_overrides(z_entry=float(preset.z_entry))
        if preset.ewma_window is not None:
            strategy = strategy.with_overrides(ewma_window=int(preset.ewma_window))
        if preset.min_sigma_to_fee_ratio is not None:
            strategy = strategy.with_overrides(
                min_sigma_to_fee_ratio=float(preset.min_sigma_to_fee_ratio)
            )
        if preset.tp_deviation_fraction is not None:
            strategy = strategy.with_overrides(
                tp_deviation_fraction=float(preset.tp_deviation_fraction)
            )
        if preset.tp_min_fee_multiple is not None:
            strategy = strategy.with_overrides(
                tp_min_fee_multiple=dec(preset.tp_min_fee_multiple)
            )
        if preset.quantity_lots is not None:
            strategy = strategy.with_overrides(quantity_lots=max(int(preset.quantity_lots), 1))
        execution = (
            replace(self.execution, topology=ExecutionTopology(preset.topology))
            if preset.topology
            else self.execution
        )
        risk = (
            replace(self.risk, active_session_windows=preset.active_session_windows)
            if preset.active_session_windows
            else self.risk
        )
        if strategy is self.strategy and execution is self.execution and risk is self.risk:
            return self
        return replace(self, strategy=strategy, execution=execution, risk=risk)

    def to_dict(self) -> Dict[str, Any]:
        preset = self.preset()
        return {
            "activePreset": preset.name,
            "tariff": self.fees.tariff.value,
            "topology": self.execution.topology.value,
            "strategy": self.strategy.to_dict(),
            "risk": {
                "maxDailyDrawdownRub": str(self.risk.max_daily_drawdown_rub),
                "marginSafetyBufferPct": self.risk.margin_safety_buffer_pct,
                "maxPositionLots": self.risk.max_position_lots,
            },
            "execution": {
                "simulatedOrderLatencyMs": self.execution.simulated_order_latency_ms,
                "queueFillFactor": self.execution.queue_fill_factor,
                "slippageSteps": self.execution.slippage_steps,
                "orderBookDepth": self.execution.order_book_depth,
            },
            "leg1": preset.leg1.to_dict(),
            "leg2": preset.leg2.to_dict(),
            "beta": preset.beta,
        }

    # ------------------------------------------------------------------ загрузка
    @staticmethod
    def load(path: str | Path | None = None) -> "AppConfig":
        config_path = Path(path or DEFAULT_CONFIG_NAME)
        if not config_path.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
        raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
        return AppConfig.from_mapping(raw, source_path=str(config_path), base_dir=str(config_path.parent.resolve()))

    @staticmethod
    def from_mapping(raw: Mapping[str, Any], source_path: Optional[str] = None, base_dir: str = ".") -> "AppConfig":
        t_invest_raw = raw.get("tInvest", {}) or {}
        logging_raw = raw.get("logging", {}) or {}
        fees_raw = raw.get("fees", {}) or {}
        risk_raw = raw.get("risk", {}) or {}
        execution_raw = raw.get("execution", {}) or {}
        storage_raw = raw.get("storage", {}) or {}
        strategy_raw = raw.get("strategy", {}) or {}

        presets: Dict[str, PairPreset] = {}
        for name, preset_raw in (raw.get("presets", {}) or {}).items():
            leg1_raw = preset_raw.get("leg1", {}) or {}
            leg2_raw = preset_raw.get("leg2", {}) or {}
            presets[name] = PairPreset(
                name=name,
                description=str(preset_raw.get("description", "")),
                leg1=InstrumentMetadata.from_dict(leg1_raw),
                leg2=InstrumentMetadata.from_dict(leg2_raw),
                beta=float(preset_raw.get("beta", 1.0)),
                beta_mode=str(preset_raw.get("betaMode", "same_underlying")),
                enabled=bool(preset_raw.get("enabled", True)),
                z_entry=(
                    float(preset_raw["zEntry"]) if preset_raw.get("zEntry") is not None else None
                ),
                topology=str(preset_raw["topology"]) if preset_raw.get("topology") else None,
                active_session_windows=(
                    str(preset_raw["activeSessionWindows"])
                    if preset_raw.get("activeSessionWindows")
                    else None
                ),
                ewma_window=(
                    int(preset_raw["ewmaWindow"]) if preset_raw.get("ewmaWindow") is not None else None
                ),
                min_sigma_to_fee_ratio=(
                    float(preset_raw["minSigmaToFeeRatio"])
                    if preset_raw.get("minSigmaToFeeRatio") is not None
                    else None
                ),
                tp_deviation_fraction=(
                    float(preset_raw["tpDeviationFraction"])
                    if preset_raw.get("tpDeviationFraction") is not None
                    else None
                ),
                tp_min_fee_multiple=(
                    float(preset_raw["tpMinFeeMultiple"])
                    if preset_raw.get("tpMinFeeMultiple") is not None
                    else None
                ),
                quantity_lots=(
                    int(preset_raw["quantityLots"])
                    if preset_raw.get("quantityLots") is not None
                    else None
                ),
            )

        active_raw = str(raw.get("activePreset", "") or "")
        if not active_raw and presets:
            active_raw = next(iter(presets.keys()))

        return AppConfig(
            t_invest=TInvestSettings(
                token=str(t_invest_raw.get("token", "") or ""),
                token_env_var=str(t_invest_raw.get("tokenEnvVar", "TINVEST_TOKEN")),
                token_file=str(t_invest_raw.get("tokenFile", "") or ""),
                app_name=str(t_invest_raw.get("appName", "BotA-Scalper")),
                request_timeout_seconds=int(t_invest_raw.get("requestTimeoutSeconds", 20)),
                account_id=str(t_invest_raw.get("accountId", "") or ""),
            ),
            logging=LoggingSettings(
                level=str(logging_raw.get("level", "Information")),
                log_dir=str(logging_raw.get("logDir", "./logs")),
                file_name=str(logging_raw.get("fileName", "bot_a.log")),
            ),
            fees=FeeSettings(
                tariff=Tariff(str(fees_raw.get("tariff", "Trader"))),
            ),
            risk=RiskSettings(
                max_daily_drawdown_rub=dec(risk_raw.get("maxDailyDrawdownRub", 15000)),
                margin_safety_buffer_pct=float(risk_raw.get("marginSafetyBufferPct", 25.0)),
                max_position_lots=int(risk_raw.get("maxPositionLots", 2)),
                trade_evening_session=bool(risk_raw.get("tradeEveningSession", True)),
                flatten_on_shutdown=bool(risk_raw.get("flattenOnShutdown", True)),
                active_session_windows=str(risk_raw.get("activeSessionWindows", "AllDay")),
            ),
            execution=ExecutionSettings(
                topology=ExecutionTopology(str(execution_raw.get("topology", "SingleLegLeadLag"))),
                simulated_order_latency_ms=int(execution_raw.get("simulatedOrderLatencyMs", 25)),
                queue_fill_factor=float(execution_raw.get("queueFillFactor", 0.35)),
                slippage_steps=int(execution_raw.get("slippageSteps", 1)),
                order_book_depth=int(execution_raw.get("orderBookDepth", 5)),
                paper_initial_cash_rub=dec(execution_raw.get("paperInitialCashRub", 300000)),
                order_poll_interval_ms=int(execution_raw.get("orderPollIntervalMs", 250)),
                cash_poll_interval_ms=int(execution_raw.get("cashPollIntervalMs", 3000)),
                stat_sample_interval_seconds=float(
                    execution_raw.get("statSampleIntervalSeconds", 5.0)
                ),
                bootstrap_warmup_days=int(execution_raw.get("bootstrapWarmupDays", 5)),
                l2_entry_persistence=bool(execution_raw.get("l2EntryPersistence", True)),
                l2_entry_confirm_samples=max(
                    int(execution_raw.get("l2EntryConfirmSamples", 2)), 1
                ),
                l2_skew_factor=float(execution_raw.get("l2SkewFactor", 1.5)),
            ),
            storage=StorageSettings(
                postgres_dsn=str(storage_raw.get("postgresDsn", "") or ""),
                recordings_dir=str(storage_raw.get("recordingsDir", "./recordings")),
                data_cache_dir=str(storage_raw.get("dataCacheDir", "./data_cache")),
                reports_dir=str(storage_raw.get("reportsDir", "./reports")),
                state_dir=str(storage_raw.get("stateDir", "./state")),
                record_to_postgres=bool(storage_raw.get("recordToPostgres", True)),
                record_to_files=bool(storage_raw.get("recordToFiles", True)),
                recording_batch_size=int(storage_raw.get("recordingBatchSize", 500)),
                recording_flush_seconds=float(storage_raw.get("recordingFlushSeconds", 2.0)),
            ),
            strategy=_align_stat_cadence(
                StrategyParams.from_mapping(strategy_raw),
                float(execution_raw.get("statSampleIntervalSeconds", 5.0)),
            ),
            presets=presets,
            active_preset=active_raw,
            source_path=source_path,
            base_dir=base_dir,
        )
