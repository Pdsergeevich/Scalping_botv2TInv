"""Rolling walk-forward движок v4.2: ротация корзины без заглядывания вперёд.

Схема (строго причинная):

1. **Сигнальные потоки** (`build_signals`): для каждой пары `(инструмент, набор движков)`
   прогоняется тот же FSM, что и в бэктесте, и сохраняются сделки «вход → выход»
   с ценами, стопом в пунктах и `conviction`. FSM решает только по закрытым барам,
   поэтому поток сигналов не содержит будущей информации.
2. **Ротация** (`select_basket`): на каждую дату `T_k` кандидаты оцениваются на
   трейлинг-окне `[T_k − L, T_k)` по **закрытым** сделкам: нужна прибыль в обеих
   половинах окна, `PF ≥ min_pf`, `≥ min_trades` сделок; затем дедупликация по
   базовому активу и выбор Top-M по Net.
3. **Исполнение** (`run_walkforward`): счёт 50 000 ₽ с общей маржой исполняет только
   сигналы активных пар, чей вход попал в срез `[T_k, T_{k+1})`. Позиция, открытая
   до ротации, закрывается своим стопом/трейлом (сигнал самодостаточен), а новые
   входы по выбывшему инструменту запрещены — как и требуется в задании.

Метрики считаются по фактической кривой equity (`equity_metrics`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from bot_a.backtest.breakout_backtest import BreakoutResult, BreakoutTrade
from bot_a.backtest.candles import Candle
from bot_a.backtest.swing_portfolio import equity_metrics
from bot_a.economics.fees import Tariff, TinkoffFeeCalculator
from bot_a.engine.breakout_fsm import Bar, BreakoutParams, BreakoutSide, BreakoutExitReason
from bot_a.engine.multiregime_fsm import regime_machine
from bot_a.models.domain import InstrumentMetadata
from bot_a.services.front_contracts import underlying_family
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import MSK

ZERO = Decimal(0)
ENGINES = ("A", "B", "C", "AB", "BC", "ABC")


def engine_params(base: BreakoutParams, engine: str) -> BreakoutParams:
    """Набор движков по коду: `A` — пробой, `B` — откаты, `C` — фейды, `AB`/`BC`/`ABC` — комбинации."""
    codes = set(str(engine).upper())
    pullback = "B" in codes
    fade = "C" in codes
    if not pullback and not fade:
        return base
    return base.overrides(
        regime_mode="multi",
        pullback_enabled=pullback,
        fade_enabled=fade,
    )


@dataclass(slots=True)
class Signal:
    ticker: str
    engine: str
    entry_ms: int
    exit_ms: int
    side: BreakoutSide
    entry_price: Decimal
    exit_price: Decimal
    stop_points: float
    conviction: float
    reason: str = ""
    family: str = ""

    @property
    def entry_date(self) -> date:
        return datetime.fromtimestamp(self.entry_ms / 1000, tz=MSK).date()

    @property
    def exit_date(self) -> date:
        return datetime.fromtimestamp(self.exit_ms / 1000, tz=MSK).date()


def build_signals(
    meta: InstrumentMetadata,
    params: BreakoutParams,
    candles: Sequence[Candle],
    engine: str,
    slip_steps: int = 1,
) -> List[Signal]:
    """Прогон FSM по истории → список самодостаточных сделок (вход/выход/стоп)."""
    machine = regime_machine(meta, engine_params(params, engine))
    slip = meta.min_price_increment * Decimal(max(int(slip_steps), 0))
    signals: List[Signal] = []
    opened: Optional[Tuple[Any, Decimal, float]] = None  # (event, entry_price, stop_points)
    for candle in candles:
        bar = Bar(
            time_ms=candle.time_ms,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=int(candle.volume),
        )
        for event in machine.on_bar(bar):
            if event.kind == "entry":
                position = machine.position
                if position is None:
                    continue
                fill = event.price + slip if event.side is BreakoutSide.LONG else event.price - slip
                stop_points = abs(float(position.entry_price - position.initial_stop))
                opened = (event, fill, stop_points)
                continue
            if event.kind != "exit" or opened is None:
                continue
            entry_event, entry_fill, stop_points = opened
            exit_fill = event.price - slip if event.side is BreakoutSide.LONG else event.price + slip
            signals.append(
                Signal(
                    ticker=meta.ticker,
                    engine=engine,
                    entry_ms=entry_event.time_ms,
                    exit_ms=event.time_ms,
                    side=event.side,
                    entry_price=entry_fill,
                    exit_price=exit_fill,
                    stop_points=stop_points,
                    conviction=float(getattr(entry_event, "conviction", 1.0) or 1.0),
                    reason=str(event.reason or ""),
                    family=underlying_family(meta.ticker),
                )
            )
            opened = None
    return signals


def build_all_signals(
    instruments: Dict[str, InstrumentMetadata],
    params_by_ticker: Dict[str, BreakoutParams],
    candles: Dict[str, Sequence[Candle]],
    engines: Iterable[str] = ENGINES,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[Tuple[str, str], List[Signal]]:
    """Сигналы по всем парам (инструмент, движки)."""
    out: Dict[Tuple[str, str], List[Signal]] = {}
    for ticker, meta in instruments.items():
        series = candles.get(ticker)
        if not series:
            continue
        params = params_by_ticker.get(ticker) or BreakoutParams()
        for engine in engines:
            out[(ticker, engine)] = build_signals(meta, params, series, engine)
        if progress is not None:
            progress(ticker)
    return out


def _window_stats(
    signals: Sequence[Signal],
    start_ms: int,
    end_ms: int,
    meta: InstrumentMetadata,
    fee_rate: float = 0.00025,
) -> Dict[str, float]:
    """Net/PF/сделки по сделкам, закрытым внутри окна (без заглядывания вперёд).

    Обязательно **после комиссии** (0.025 % Премиум за сторону от нотионала):
    отбор по валовому PnL выбирает сделки, где край съедается издержками.
    """
    selected = [item for item in signals if start_ms <= item.exit_ms < end_ms]
    if not selected:
        return {"net": 0.0, "pf": 0.0, "trades": 0}
    rub = float(meta.rub_per_price_unit)
    net = 0.0
    profits = 0.0
    losses = 0.0
    for item in selected:
        sign = 1.0 if item.side is BreakoutSide.LONG else -1.0
        gross = sign * float(item.exit_price - item.entry_price) * rub
        fee = fee_rate * float(
            meta.value_in_rub(item.entry_price) + meta.value_in_rub(item.exit_price)
        )
        value = gross - fee
        net += value
        if value > 0:
            profits += value
        else:
            losses -= value
    return {
        "net": net,
        "pf": (profits / losses) if losses > 0 else (10.0 if profits > 0 else 0.0),
        "trades": len(selected),
    }


def select_basket(
    signals: Dict[Tuple[str, str], List[Signal]],
    metas: Dict[str, InstrumentMetadata],
    at_ms: int,
    lookback_days: int = 60,
    top: int = 8,
    min_pf: float = 1.30,
    min_trades: int = 4,
    deduplicate: bool = True,
) -> List[Dict[str, Any]]:
    """Кандидаты на трейлинг-окне `[at_ms − L, at_ms)`: устойчивость + дедуп + Top-M."""
    start_ms = at_ms - int(lookback_days) * 86_400_000
    middle_ms = (start_ms + at_ms) // 2
    scored: List[Dict[str, Any]] = []
    for key, series in signals.items():
        ticker, engine = key if isinstance(key, tuple) else (str(key), "A")
        meta = metas.get(ticker)
        if meta is None:
            continue
        first = _window_stats(series, start_ms, middle_ms, meta)
        second = _window_stats(series, middle_ms, at_ms, meta)
        full = _window_stats(series, start_ms, at_ms, meta)
        if full["trades"] < min_trades:
            continue
        if first["net"] <= 0 or second["net"] <= 0:
            continue
        if full["pf"] < min_pf:
            continue
        scored.append(
            {
                "ticker": ticker,
                "engine": engine,
                "net": full["net"],
                "pf": full["pf"],
                "trades": full["trades"],
                "family": underlying_family(ticker),
            }
        )
    scored.sort(key=lambda item: (-item["net"], item["ticker"], item["engine"]))
    if not deduplicate:
        return scored[:top]
    kept: List[Dict[str, Any]] = []
    used: set = set()
    for item in scored:
        if item["family"] in used:
            continue
        kept.append(item)
        used.add(item["family"])
        if len(kept) >= top:
            break
    return kept


@dataclass(slots=True)
class WalkForwardSettings:
    capital_rub: float = 50_000.0
    risk_pct_per_trade: float = 0.7
    tariff: Tariff = Tariff.PREMIUM
    margin_safety_buffer_pct: float = 25.0
    max_margin_utilization_pct: float = 75.0
    max_concurrent_positions: int = 3
    lookback_days: int = 60
    step_days: int = 14
    top: int = 8
    min_pf: float = 1.30
    min_trades: int = 4
    conviction_sizing: bool = False
    drawdown_throttle: bool = True
    throttle_start_pct: float = 5.0
    throttle_min_multiplier: float = 0.4
    deduplicate: bool = True


@dataclass(slots=True)
class WalkForwardResult:
    trades: List[BreakoutTrade] = field(default_factory=list)
    equity_curve: List[Tuple[int, float]] = field(default_factory=list)
    baskets: List[Dict[str, Any]] = field(default_factory=list)
    max_margin_used: float = 0.0
    rejected_by_slots: int = 0
    rejected_by_risk: int = 0
    rejected_by_margin: int = 0

    def metrics(self, capital: float) -> Dict[str, Any]:
        return equity_metrics(self.equity_curve, self.trades, capital)


def _throttle(settings: WalkForwardSettings, equity: float, peak: float) -> float:
    if not settings.drawdown_throttle or peak <= 0:
        return 1.0
    drawdown = max(peak - equity, 0.0) / peak * 100.0
    start = max(float(settings.throttle_start_pct), 0.1)
    if drawdown <= start:
        return 1.0
    scale = min((drawdown - start) / max(30.0 - start, 1.0), 1.0)
    floor = max(min(float(settings.throttle_min_multiplier), 1.0), 0.1)
    return 1.0 - (1.0 - floor) * scale


def run_walkforward(
    signals: Dict[Tuple[str, str], List[Signal]],
    metas: Dict[str, InstrumentMetadata],
    start_ms: int,
    end_ms: int,
    settings: Optional[WalkForwardSettings] = None,
) -> WalkForwardResult:
    """Ротация + исполнение на общем счёте; метрики — по фактической equity.

    На каждом срезе `[T_k, T_{k+1})` активны только пары из `select_basket(T_k)`.
    Сделка самодостаточна (вход и выход внутри сигнала), поэтому позиция, открытая
    до ротации, доигрывается своим стопом/трейлом, а новые входы по выбывшему
    инструменту не открываются.
    """
    settings = settings or WalkForwardSettings()
    result = WalkForwardResult()
    fees = TinkoffFeeCalculator(settings.tariff)
    equity = float(settings.capital_rub)
    peak = equity
    step_ms = max(int(settings.step_days), 1) * 86_400_000
    lookback = max(int(settings.lookback_days), 1)
    rebalance_ms = start_ms + lookback * 86_400_000

    def open_trades(at_ms: int) -> List[BreakoutTrade]:
        return [trade for trade in result.trades if trade.entry_ms <= at_ms < trade.exit_ms]

    def reserved_margin(at_ms: int) -> float:
        total = 0.0
        for trade in open_trades(at_ms):
            meta = metas.get(trade.ticker)
            if meta is None:
                continue
            total += margin_rate(meta) * trade.quantity
        return total

    def margin_rate(meta: InstrumentMetadata) -> float:
        return float(meta.position_margin_rub(1)) * (
            1.0 + float(settings.margin_safety_buffer_pct) / 100.0
        )

    while rebalance_ms <= end_ms:
        basket = select_basket(
            signals,
            metas,
            rebalance_ms,
            lookback_days=lookback,
            top=settings.top,
            min_pf=settings.min_pf,
            min_trades=settings.min_trades,
            deduplicate=settings.deduplicate,
        )
        result.baskets.append(
            {
                "date": datetime.fromtimestamp(rebalance_ms / 1000, tz=MSK).date().isoformat(),
                "pairs": [
                    {
                        "ticker": item["ticker"],
                        "engine": item["engine"],
                        "net": round(item["net"], 2),
                        "pf": round(item["pf"], 2),
                        "trades": item["trades"],
                    }
                    for item in basket
                ],
            }
        )
        slice_end = min(rebalance_ms + step_ms, end_ms)
        executions: List[Signal] = []
        for item in basket:
            key = (item["ticker"], item["engine"])
            executions.extend(
                signal for signal in signals.get(key, []) if rebalance_ms <= signal.entry_ms < slice_end
            )
        executions.sort(key=lambda signal: (signal.entry_ms, signal.ticker, signal.engine))
        for signal in executions:
            meta = metas.get(signal.ticker)
            if meta is None:
                continue
            margin_used = reserved_margin(signal.entry_ms)
            if (
                settings.max_concurrent_positions > 0
                and len(open_trades(signal.entry_ms)) >= settings.max_concurrent_positions
            ):
                result.rejected_by_slots += 1
                continue
            throttle = _throttle(settings, equity, peak)
            conviction = signal.conviction if settings.conviction_sizing else 1.0
            risk_budget = max(equity, 0.0) * float(settings.risk_pct_per_trade) / 100.0
            stop_rub = max(signal.stop_points * float(meta.rub_per_price_unit), 1e-9)
            quantity = int(risk_budget * conviction * throttle // stop_rub)
            if quantity <= 0:
                result.rejected_by_risk += 1
                continue
            util_cap = max(equity, 0.0) * float(settings.max_margin_utilization_pct) / 100.0
            free = max(min(util_cap, equity) - margin_used, 0.0)
            rate = margin_rate(meta)
            by_margin = int(free // rate) if rate > 0 else quantity
            if by_margin < quantity:
                quantity = by_margin
                if quantity <= 0:
                    result.rejected_by_margin += 1
                    continue
            notional_entry = meta.value_in_rub(signal.entry_price) * Decimal(quantity)
            notional_exit = meta.value_in_rub(signal.exit_price) * Decimal(quantity)
            fees.roll_day(signal.entry_date)
            entry_fee = fees.register_notional(notional_entry, signal.ticker)
            fees.roll_day(signal.exit_date)
            exit_fee = fees.register_notional(notional_exit, signal.ticker)
            sign = Decimal(1) if signal.side is BreakoutSide.LONG else Decimal(-1)
            gross = float(
                (signal.exit_price - signal.entry_price) * sign * meta.rub_per_price_unit
                * Decimal(quantity)
            )
            total_fees = float(entry_fee + exit_fee)
            trade = BreakoutTrade(
                ticker=signal.ticker,
                side=signal.side,
                quantity=quantity,
                entry_ms=signal.entry_ms,
                exit_ms=signal.exit_ms,
                entry_price=signal.entry_price,
                exit_price=signal.exit_price,
                gross_rub=gross,
                fees_rub=total_fees,
                net_rub=gross - total_fees,
                reason=BreakoutExitReason.HARD_STOP,
                atr_points=0.0,
                mfe_points=0.0,
                hold_minutes=(signal.exit_ms - signal.entry_ms) / 60_000.0,
            )
            result.trades.append(trade)
            equity += trade.net_rub
            peak = max(peak, equity)
            result.max_margin_used = max(result.max_margin_used, margin_used + rate * quantity)
            result.equity_curve.append((signal.exit_ms, equity))
        rebalance_ms += step_ms
    result.equity_curve.sort(key=lambda item: item[0])
    return result


def update_appsettings(
    path: str | Path,
    selection: Dict[str, str],
    portfolio: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Записывает отобранную корзину и движки в `appsettings.json` (v4.2).

    Меняет только `swingBasket`, `swingPortfolio` и флаги движков в пресетах —
    остальные настройки файла сохраняются как есть.
    """
    target = Path(path)
    config = json.loads(target.read_text(encoding="utf-8"))
    presets = config.setdefault("breakoutPresets", {})
    for ticker, engine in selection.items():
        preset = presets.setdefault(ticker, {"description": f"v4.2 walk-forward {ticker}"})
        codes = set(str(engine).upper())
        preset["regimeMode"] = "multi" if (codes - {"A"}) else "breakout"
        preset["pullbackEnabled"] = "B" in codes
        preset["fadeEnabled"] = "C" in codes
        preset["description"] = f"v4.2 walk-forward {ticker} · движки {engine}"
    config["swingBasket"] = list(selection)
    if portfolio is not None:
        config["swingPortfolio"] = dict(portfolio)
    target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config


def save_report(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
