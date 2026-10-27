"""Live-дашборд: живой стакан T-Invest, метрики ядра и сделки в браузере.

Запуск::

    py -3 -m bot_a --mode watch --preset VTBR_SBER --port 8099

Сервис опрашивает ``MarketDataService/GetOrderBook`` (глубина 20) с интервалом
100–500 мс, кормит **то же** ядро, что и paper/live, и публикует состояние по
HTTP: ``/`` — страница с автообновлением, ``/api/state`` — JSON-снимок.

Ничего не покупает и не продаёт: исполнение идёт через
``SimulatedOrderExecutionGateway`` (реальные деньги не тратятся).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from bot_a.broker.tinkoff_rest_client import TInvestRestClient, TInvestRestError
from bot_a.config import AppConfig, PairPreset
from bot_a.economics.fees import TinkoffFeeCalculator
from bot_a.engine.factory import create_engine, create_guard
from bot_a.engine.fsm import StrategyParams
from bot_a.engine.ledger import PortfolioLedger
from bot_a.execution.simulated_gateway import SimulatedOrderExecutionGateway
from bot_a.models.domain import ExecutionTopology
from bot_a.services.l2_sync import PairBookSyncer
from bot_a.utils.time_provider import MSK, RealTimeProvider, ms_to_msk

LOGGER_NAME = "bot_a.watch"
DEFAULT_PORT = 8099


def _f(value: Any) -> Optional[float]:
    """Decimal/None -> float для JSON."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _DayTracker:
    """Итоги текущего торгового дня: как близко Z подходил к порогу и что мешало.

    Живёт в памяти и раз в минуту дописывает строку в ``state/watch_<пресет>_<дата>.jsonl``,
    поэтому вопрос «почему за день не было входов» разбирается по факту, а не по памяти.
    """

    def __init__(self, preset: str, state_dir: Path) -> None:
        self._preset = preset
        self._state_dir = Path(state_dir)
        self._day: Optional[str] = None
        self._zmin = 0.0
        self._zmax = 0.0
        self._max_abs = 0.0
        self._max_at = ""
        self._samples = 0
        self._over = 0
        self._ready = 0
        self._started = ""
        self._last_dump_ms = 0
        self._path: Optional[Path] = None

    def _reset(self, day: str, stamp: str) -> None:
        self._day = day
        self._zmin = self._zmax = self._max_abs = 0.0
        self._max_at = ""
        self._samples = self._over = self._ready = 0
        self._started = stamp
        self._last_dump_ms = 0
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._path = self._state_dir / f"watch_{self._preset}_{day}.jsonl"
        except OSError:
            self._path = None

    def feed(self, metrics: Any, ready: bool, z_entry: float) -> None:
        if metrics is None:
            return
        z = float(metrics.zscore)
        ts = int(metrics.time_ms)
        stamp = ms_to_msk(ts)
        day = stamp.strftime("%Y-%m-%d")
        if day != self._day:
            self._reset(day, stamp.strftime("%H:%M:%S"))
        self._samples += 1
        self._zmin = min(self._zmin, z)
        self._zmax = max(self._zmax, z)
        if abs(z) >= z_entry:
            self._over += 1
        if ready:
            self._ready += 1
        if abs(z) > self._max_abs:
            self._max_abs = abs(z)
            self._max_at = stamp.strftime("%H:%M:%S")
        if self._path is not None and ts - self._last_dump_ms >= 5_000:
            self._last_dump_ms = ts
            line = {
                "t": stamp.isoformat(),
                "z": round(z, 4),
                "s": round(float(metrics.spread_rub), 5),
                "mu": round(float(metrics.mean_rub), 5),
                "sigma": round(float(metrics.sigma_rub), 5),
                "ready": bool(ready),
                "obi1": round(float(metrics.obi_leg1), 3),
            }
            try:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(line, ensure_ascii=False) + "\n")
            except OSError:
                self._path = None

    def payload(self, z_entry: float) -> Dict[str, Any]:
        return {
            "date": self._day,
            "started": self._started,
            "samples": self._samples,
            "zMin": round(self._zmin, 3),
            "zMax": round(self._zmax, 3),
            "maxAbs": round(self._max_abs, 3),
            "maxAt": self._max_at,
            "zEntry": z_entry,
            "aboveThreshold": self._over,
            "readySamples": self._ready,
            "file": None if self._path is None else str(self._path),
        }


class DashboardState:
    """Потокобезопасный снимок состояния для HTTP-слоя."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._payload: Dict[str, Any] = {"status": "starting"}

    def publish(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._payload = payload

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._payload


class _Handler(BaseHTTPRequestHandler):
    state: DashboardState
    page: bytes = b""

    def log_message(self, fmt: str, *args: Any) -> None:  # тише в консоли
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/state"):
            body = json.dumps(self.state.snapshot(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(self.page)))
        self.end_headers()
        self.wfile.write(self.page)


class LiveWatchService:
    """Живой стакан + ядро бота + HTTP-дашборд."""

    def __init__(
        self,
        config: AppConfig,
        preset: PairPreset,
        client: TInvestRestClient,
        params: Optional[StrategyParams] = None,
        topology: Optional[ExecutionTopology] = None,
        interval_ms: int = 300,
        depth: int = 20,
        port: int = DEFAULT_PORT,
        history: int = 900,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._preset = preset
        self._client = client
        base = params or config.strategy
        self._params = replace(
            base,
            beta=float(preset.beta),
            beta_mode=(preset.beta_mode if base.beta_mode in ("", "auto") else base.beta_mode),
            stat_sample_interval_seconds=float(config.execution.stat_sample_interval_seconds),
            use_entry_persistence=bool(config.execution.l2_entry_persistence),
            entry_confirm_samples=int(config.execution.l2_entry_confirm_samples),
        )
        self._topology = topology or config.execution.topology
        self._interval_ms = max(int(interval_ms), 100)
        self._depth = max(int(depth), 5)
        self._port = int(port)
        self._log = logger or logging.getLogger(LOGGER_NAME)
        self._state = DashboardState()
        self._history: Deque[Dict[str, float]] = deque(maxlen=max(int(history), 60))
        self._prints: Deque[Dict[str, Any]] = deque(maxlen=400)
        self._seen_trades: set = set()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._day = _DayTracker(self._preset.name, config.path(config.storage.state_dir))
        self._syncer = PairBookSyncer(
            interval_ms=self._interval_ms,
            skew_factor=float(config.execution.l2_skew_factor),
        )

    # ------------------------------------------------------------------ HTTP
    def _start_server(self) -> str:
        _Handler.state = self._state
        _Handler.page = DASHBOARD_HTML.encode("utf-8")
        self._server = ThreadingHTTPServer(("127.0.0.1", self._port), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        url = f"http://127.0.0.1:{self._port}/"
        self._log.info("Дашборд доступен: %s", url)
        return url

    def _stop_server(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # ------------------------------------------------------------------ данные
    def _entry_diagnostics(self, engine: Any, metrics: Any) -> Dict[str, Any]:
        """Почему вход разрешён/запрещён прямо сейчас (зеркало фильтров ФСМ).

        Только читает состояние ядра: ничего не меняет и не отправляет заявок.
        """
        params = engine.params
        if metrics is None:
            return {"ready": False, "missing": ["нет метрик (стакан ещё не собран)"], "checks": {}}
        z = float(metrics.zscore)
        az = abs(z)
        long_side = z >= 0
        side = "buy" if long_side else "sell"
        checks: Dict[str, bool] = {}
        details: Dict[str, str] = {}

        checks["z"] = az >= params.z_entry
        details["z"] = f"|Z| {az:.2f} при пороге {params.z_entry:.2f}"

        try:
            hook_ok = (not params.use_z_peak_hook) or engine._z_peak_confirmed(z)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            hook_ok = True
        checks["hook"] = hook_ok
        details["hook"] = (
            f"пик {getattr(engine, '_z_peak', 0.0):.2f}, откат "
            f"{max(getattr(engine, '_z_peak', 0.0) - az, 0.0):.2f} (нужно ≥ {params.z_hook_pullback:.2f})"
        )

        try:
            regime_ok = engine._regime_confirmed(z)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            regime_ok = True
        checks["regime"] = regime_ok
        details["regime"] = "Leg2 ведёт / Leg1 изолирован" if regime_ok else "ноги идут трендом вместе"

        obi_ok = (
            metrics.obi_leg1 > -params.obi_entry_guard
            if long_side
            else metrics.obi_leg1 < params.obi_entry_guard
        )
        checks["obi"] = obi_ok
        details["obi"] = f"OBI Leg1 {metrics.obi_leg1:+.2f} (порог ±{params.obi_entry_guard:.2f})"

        try:
            fee = engine._current_round_trip_fee(params.quantity_lots)  # noqa: SLF001
            sigma_leg1 = engine._sigma_leg1_rub(metrics)  # noqa: SLF001
            vol_ok = (
                not params.use_vol_to_fee_filter
                or float(sigma_leg1) >= float(fee) * params.min_sigma_to_fee_ratio
            )
            details["vol"] = (
                f"σ_leg1 {float(sigma_leg1):.4f} / Fee {float(fee):.4f} = "
                f"{(float(sigma_leg1) / float(fee)) if float(fee) else 0:.2f} "
                f"(нужно ≥ {params.min_sigma_to_fee_ratio:.2f})"
            )
        except Exception:  # noqa: BLE001
            fee = None
            vol_ok = True
            details["vol"] = "оценка недоступна"
        checks["vol"] = vol_ok

        try:
            expected = TinkoffFeeCalculator.expected_gross_reversion_rub(
                metrics.spread_rub, metrics.mean_rub, z, params.z_exit
            )
            if engine._topology is ExecutionTopology.SINGLE_LEG_LEAD_LAG and metrics.beta > 1e-9:  # noqa: SLF001
                # v2.6.1: как и в ядре, приводим ожидаемый возврат к рублям Leg1.
                expected = Decimal(expected) / Decimal(str(metrics.beta))
            slippage = TinkoffFeeCalculator.slippage_buffer_rub(
                engine._leg1, params.slippage_buffer_steps, params.quantity_lots  # noqa: SLF001
            )
            edge_ok = fee is not None and TinkoffFeeCalculator.is_edge_sufficient(
                expected, fee, slippage, params.min_edge_multiplier
            )
            details["edge"] = (
                f"ожидаемый возврат {float(expected):.3f} RUB против "
                f"{params.min_edge_multiplier}×({float(fee or 0):.3f}+{float(slippage):.3f})"
            )
        except Exception:  # noqa: BLE001
            edge_ok = True
            details["edge"] = "оценка недоступна"
        checks["edge"] = edge_ok

        try:
            window_ok = engine._guard.entry_window_reason(int(metrics.time_ms)) is None  # noqa: SLF001
        except Exception:  # noqa: BLE001
            window_ok = True
        checks["window"] = window_ok
        details["window"] = "окно входа открыто" if window_ok else "вне активного окна (PrimeHoursOnly)"

        try:
            daily_ok = engine.daily_locked_reason is None
            details["daily"] = engine.daily_locked_reason or "дневной предохранитель не сработал"
        except Exception:  # noqa: BLE001
            daily_ok = True
            details["daily"] = "—"
        checks["daily"] = daily_ok

        missing = [
            {
                "z": "|Z| ниже порога входа",
                "hook": "нет подтверждения разворота (хук пика Z)",
                "regime": "ноги движутся одним трендом",
                "obi": "поток стакана против входа",
                "vol": "волатильность слишком мала против комиссии",
                "edge": "ожидаемый возврат не покрывает комиссию",
                "window": "вне окна входа",
                "daily": "сработал дневной предохранитель",
            }[name]
            for name, ok in checks.items()
            if not ok
        ]
        return {
            "ready": all(checks.values()),
            "side": side,
            "z": round(z, 3),
            "distance": round(max(params.z_entry - az, 0.0), 3),
            "progressPct": round(min(az / params.z_entry * 100.0, 999.0), 1),
            "checks": checks,
            "details": details,
            "missing": missing,
        }

    def _book_payload(self, book: Any, levels: int = 10) -> Dict[str, Any]:
        bids = [{"p": _f(level.price), "q": level.quantity} for level in book.bids[:levels]]
        asks = [{"p": _f(level.price), "q": level.quantity} for level in book.asks[:levels]]
        return {
            "bids": bids,
            "asks": asks,
            "bestBid": _f(book.bids[0].price) if book.bids else None,
            "bestAsk": _f(book.asks[0].price) if book.asks else None,
            "t": book.time_ms,
        }

    def _publish(
        self,
        engine: Any,
        ledger: PortfolioLedger,
        books: Dict[int, Any],
        started: float,
        snapshots: int,
        warmup: Dict[str, Any],
        rest_errors: int,
    ) -> None:
        metrics = engine.metrics
        position = engine.position
        snapshot = engine.snapshot()
        now = time.monotonic()
        entry = self._entry_diagnostics(engine, metrics)
        if metrics is not None:
            self._day.feed(metrics, bool(entry["ready"]), float(self._params.z_entry))
            self._history.append(
                {
                    "t": metrics.time_ms,
                    "z": float(metrics.zscore),
                    "s": float(metrics.spread_rub),
                    "m1": float(metrics.micro_leg1),
                    "m2": float(metrics.micro_leg2),
                    "ready": 1.0 if entry["ready"] else 0.0,
                }
            )
        trades = [
            {
                "id": trade.trade_id,
                "openMsk": ms_to_msk(trade.opened_ms).strftime("%H:%M:%S"),
                "closeMsk": ms_to_msk(trade.closed_ms).strftime("%H:%M:%S"),
                "side": trade.side_leg1.value,
                "entry": _f(trade.leg1_entry_price),
                "exit": _f(trade.leg1_exit_price),
                "reason": trade.exit_reason.value,
                "entryZ": round(trade.entry_z, 2),
                "exitZ": round(trade.exit_z, 2),
                "gross": _f(trade.gross_pnl_rub),
                "fees": _f(trade.fees_rub),
                "net": _f(trade.net_pnl_rub),
                "holdMin": round(trade.hold_seconds / 60.0, 1),
            }
            for trade in ledger.trades[-25:]
        ]
        totals = ledger.totals()
        unrealized = None
        if position is not None and metrics is not None:
            try:
                unrealized = _f(engine._leg1_unrealized_gross(position, metrics))  # noqa: SLF001
            except Exception:  # noqa: BLE001 — диагностика не должна ломать поток
                unrealized = None
        payload = {
            "status": "live",
            "serverTimeMsk": datetime.now(tz=MSK).strftime("%H:%M:%S"),
            "preset": self._preset.name,
            "leg1": self._preset.leg1.ticker,
            "leg2": self._preset.leg2.ticker,
            "topology": self._topology.value,
            "tariff": self._config.fees.tariff.value,
            "params": {
                "zEntry": self._params.z_entry,
                "zExit": self._params.z_exit,
                "zStop": self._params.z_stop,
                "ewma": self._params.ewma_window,
                "maxHoldMin": round(self._params.max_hold_seconds / 60.0, 1),
                "sampleSec": self._params.stat_sample_interval_seconds,
                "windows": self._config.risk.active_session_windows,
            },
            "metrics": None
            if metrics is None
            else {
                "z": round(metrics.zscore, 3),
                "spread": round(metrics.spread_rub, 5),
                "mean": round(metrics.mean_rub, 5),
                "sigma": round(metrics.sigma_rub, 5),
                "sigmaLeg1": round(float(engine._sigma_leg1_rub(metrics)), 5),  # noqa: SLF001
                "beta": round(metrics.beta, 5),
                "betaMacro": round(metrics.beta_macro, 5),
                "betaDriftPct": None
                if metrics.beta_drift is None
                else round(metrics.beta_drift * 100.0, 4),
                "obi1": round(metrics.obi_leg1, 3),
                "obi2": round(metrics.obi_leg2, 3),
                "micro1": _f(metrics.micro_leg1),
                "micro2": _f(metrics.micro_leg2),
                "ticks": metrics.tick_count,
                "isWarm": metrics.is_warm,
                "sampled": metrics.stat_sampled,
                "timeMsk": ms_to_msk(metrics.time_ms).strftime("%H:%M:%S"),
            },
            "books": {
                "leg1": self._book_payload(books[1]) if 1 in books else None,
                "leg2": self._book_payload(books[2]) if 2 in books else None,
            },
            "bot": {
                "state": snapshot["state"],
                "position": None
                if position is None
                else {
                    "side": position.side_leg1.value,
                    "entry": _f(position.leg1_entry_price),
                    "qty": position.quantity,
                    "entryZ": round(position.entry_z, 2),
                    "unrealized": unrealized,
                    "openedMsk": ms_to_msk(position.opened_ms).strftime("%H:%M:%S"),
                },
                "trades": snapshot["trades"],
                "netRub": _f(snapshot["netPnlRub"]),
                "feesRub": _f(snapshot["feesRub"]),
                "dailyLocked": snapshot.get("dailyLockedReason"),
                "counters": snapshot.get("counters", {}),
                "tradeList": trades,
                "winRate": totals.win_rate_pct,
                "profitFactor": totals.profit_factor,
                "maxDrawdown": _f(totals.max_drawdown_rub),
            },
            "history": list(self._history)[-600:],
            "entry": entry,
            "day": self._day.payload(float(self._params.z_entry)),
            "prints": sorted(self._prints, key=lambda item: item["timeMsk"])[-40:],
            "printCounts": {
                self._preset.leg1.ticker: sum(
                    1 for p in self._prints if p["ticker"] == self._preset.leg1.ticker
                ),
                self._preset.leg2.ticker: sum(
                    1 for p in self._prints if p["ticker"] == self._preset.leg2.ticker
                ),
            },
            "runtime": {
                "seconds": round(now - started, 1),
                "snapshots": snapshots,
                "restErrors": rest_errors,
                "skewSkips": self._syncer.skew_skips,
                "lastSkewMs": round(self._syncer.last_skew_ms, 1),
                "intervalMs": self._interval_ms,
                "warmup": warmup,
            },
        }
        self._state.publish(payload)

    # ------------------------------------------------------------------ сессия
    async def run(
        self,
        duration_seconds: Optional[float] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        clock = RealTimeProvider()
        legs = [self._preset.leg1, self._preset.leg2]
        fee_calculator = TinkoffFeeCalculator(self._config.fees.tariff, clock=clock)
        gateway = SimulatedOrderExecutionGateway(
            metadata=legs,
            clock=clock,
            fee_calculator=fee_calculator,
            latency_ms=self._config.execution.simulated_order_latency_ms,
            queue_fill_factor=self._config.execution.queue_fill_factor,
            slippage_steps=self._config.execution.slippage_steps,
            order_book_depth=self._config.execution.order_book_depth,
            initial_cash_rub=self._config.execution.paper_initial_cash_rub,
        )
        guard = create_guard(self._config, legs)
        ledger = PortfolioLedger(self._config.execution.paper_initial_cash_rub)
        engine = create_engine(
            legs,
            self._params,
            self._topology,
            gateway,
            fee_calculator,
            clock,
            guard,
            ledger,
            self._log,
            daily_state_path=str(
                self._config.path(self._config.storage.state_dir)
                / f"daily_state_{self._preset.name}.json"
            ),
        )

        warmup: Dict[str, Any] = {}
        days = int(self._config.execution.bootstrap_warmup_days)
        if days > 0:
            from bot_a.services.bootstrap import bootstrap_warmup

            self._state.publish({"status": "warmup", "message": "прогрев из MOEX ISS…"})
            try:
                report = await bootstrap_warmup(
                    self._config, self._preset, engine, days=days, logger=self._log
                )
                warmup = report.to_dict()
            except Exception as exc:  # noqa: BLE001
                self._log.warning("Прогрев не удался (%s): %s", type(exc).__name__, exc)
                engine.set_bootstrap(False)

        url = self._start_server()
        await gateway.start()
        self._log.info(
            "Live-дашборд: %s, стакан %d мс, глубина %d → %s",
            self._preset.name,
            self._interval_ms,
            self._depth,
            url,
        )
        started = time.monotonic()
        deadline = (started + duration_seconds) if duration_seconds else None
        books: Dict[int, Any] = {}
        fresh: Dict[int, Any] = {}
        fetched_at: Dict[int, float] = {}
        failed_legs: List[int] = []
        snapshots = 0
        rest_errors = 0
        last_trades_poll = 0.0
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    self._log.info("Длительность сессии истекла")
                    break
                for leg, meta in ((1, legs[0]), (2, legs[1])):
                    try:
                        book = await self._client.get_order_book(meta.instrument_uid, self._depth)
                    except TInvestRestError as exc:
                        rest_errors += 1
                        failed_legs.append(leg)
                        if rest_errors % 20 == 1:
                            self._log.warning("GetOrderBook %s: %s", meta.ticker, exc)
                        continue
                    except Exception as exc:  # noqa: BLE001 — сеть не должна ронять дашборд
                        rest_errors += 1
                        failed_legs.append(leg)
                        if rest_errors % 20 == 1:
                            self._log.warning(
                                "GetOrderBook %s: неожиданная ошибка %s: %s",
                                meta.ticker,
                                type(exc).__name__,
                                exc,
                            )
                        continue
                    if book is None:
                        failed_legs.append(leg)
                        continue
                    books[leg] = book
                    fresh[leg] = book
                    fetched_at[leg] = time.monotonic() * 1000.0
                decision = self._syncer.check(fresh, fetched_at, failed_legs)
                if decision.ok:
                    for leg in (1, 2):
                        await engine.on_order_book(leg, fresh[leg])
                        snapshots += 1
                elif self._syncer.skew_skips % 50 == 1:
                    self._log.warning(
                        "Рассинхрон стаканов (%d): %s — тик пропущен для Z",
                        self._syncer.skew_skips,
                        decision.reason,
                    )
                fresh.clear()
                fetched_at.clear()
                failed_legs.clear()
                now = time.monotonic()
                if now - last_trades_poll >= 1.0:
                    last_trades_poll = now
                    try:
                        await self._poll_prints(engine, legs)
                    except Exception as exc:  # noqa: BLE001
                        rest_errors += 1
                        self._log.warning("GetLastTrades: %s: %s", type(exc).__name__, exc)
                await engine.on_clock()
                try:
                    self._publish(engine, ledger, books, started, snapshots, warmup, rest_errors)
                except Exception as exc:  # noqa: BLE001 — публикация не должна ломать сессию
                    self._log.warning("Публикация состояния: %s: %s", type(exc).__name__, exc)
                await asyncio.sleep(self._interval_ms / 1000.0)
        except KeyboardInterrupt:
            self._log.warning("Прервано пользователем")
        finally:
            await engine.shutdown(flatten=self._config.risk.flatten_on_shutdown)
            await gateway.stop()
            self._stop_server()
        return engine.snapshot()

    async def _poll_prints(self, engine: Any, legs: List[Any]) -> None:
        since = datetime.now(tz=timezone.utc) - timedelta(seconds=60)
        for leg, meta in ((1, legs[0]), (2, legs[1])):
            try:
                trades = await self._client.get_last_trades(meta.instrument_uid, since)
            except TInvestRestError:
                continue
            for trade in trades:
                key = (meta.instrument_uid, trade.time_ms, str(trade.price), trade.quantity)
                if key in self._seen_trades:
                    continue
                self._seen_trades.add(key)
                await engine.on_trade(leg, trade)
                self._prints.append(
                    {
                        "leg": leg,
                        "ticker": meta.ticker,
                        "timeMsk": ms_to_msk(trade.time_ms).strftime("%H:%M:%S"),
                        "price": _f(trade.price),
                        "qty": trade.quantity,
                        "side": trade.side.value if trade.side else "",
                    }
                )
        if len(self._seen_trades) > 50_000:
            self._seen_trades.clear()


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>Bot A — живой стакан</title>
<style>
:root{--bg:#0b1016;--panel:#131b24;--line:#22303f;--text:#dbe6f2;--muted:#7d8fa3;
--up:#26d07c;--down:#ff5b6e;--accent:#4fa3ff;--warn:#f5a623;--z:#c678dd}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:13px/1.35 'Segoe UI',system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;gap:14px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line)}
h1{font-size:15px;margin:0;font-weight:600}
.badge{padding:3px 9px;border-radius:11px;background:#1b2734;border:1px solid var(--line);font-size:12px}
.state{background:#12351f;border-color:#1f6b3f;color:#7ee2a8}
.state.bad{background:#3a1620;border-color:#7a2431;color:#ff9aa7}
.grid{display:grid;grid-template-columns:1fr 1fr 1.15fr;gap:12px;padding:12px 16px}
@media(max-width:1200px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:10px}
.panel h2{font-size:12px;margin:0 0 8px;color:var(--muted);font-weight:600;letter-spacing:.4px;text-transform:uppercase}
.row{display:flex;justify-content:space-between;gap:8px;font-variant-numeric:tabular-nums}
.kv{display:grid;grid-template-columns:repeat(2,1fr);gap:4px 12px;font-variant-numeric:tabular-nums}
.kv div{display:flex;justify-content:space-between;gap:8px}
.kv b{font-weight:600}
.book{font-variant-numeric:tabular-nums}
.lvl{display:grid;grid-template-columns:1fr 1fr 62px;gap:4px;padding:1px 2px;position:relative;font-size:12px}
.lvl .bar{position:absolute;right:74px;top:1px;bottom:1px;opacity:.28;border-radius:2px}
.lvl.bid .bar{background:var(--up)}.lvl.ask .bar{background:var(--down)}
.lvl .q{color:var(--muted);text-align:right}
.lvl.best{outline:1px solid var(--line);border-radius:4px}
.midline{display:flex;justify-content:space-between;padding:5px 2px;margin:4px 0;border-top:1px dashed var(--line);border-bottom:1px dashed var(--line)}
.zbig{font-size:30px;font-weight:700;color:var(--z);font-variant-numeric:tabular-nums}
.gauge{position:relative;height:26px;background:#0f1620;border:1px solid var(--line);border-radius:6px;margin:8px 0 4px}
.gauge .tick{position:absolute;top:0;bottom:0;width:1px;background:#2c3b4d}
.gauge .mark{position:absolute;top:-3px;width:3px;height:32px;background:var(--accent);border-radius:2px;box-shadow:0 0 8px var(--accent)}
.gauge .zone{position:absolute;top:0;bottom:0;background:rgba(79,163,255,.1)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:3px 5px;text-align:right;border-bottom:1px solid #1a2530;font-size:12px}
th{color:var(--muted);font-weight:600;text-align:right}
td.l,th.l{text-align:left}
.pos{color:var(--up)}.neg{color:var(--down)}
svg{width:100%;height:120px;display:block}
#chart{height:270px}
.note{color:var(--muted);font-size:11px;margin-top:6px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--up);margin-right:6px;animation:p 1.4s infinite}
@keyframes p{0%{opacity:.25}50%{opacity:1}100%{opacity:.25}}
</style></head><body>
<header>
 <h1><span class="dot"></span>Bot A — живой стакан</h1>
 <div class="badge" id="preset">—</div>
 <div class="badge" id="stateBadge">—</div>
 <div class="badge" id="mode">—</div>
 <div class="badge" id="clock">—</div>
 <div class="badge" id="stats">—</div>
</header>
<div class="grid">
 <div class="panel">
  <h2>Стакан <span id="leg1name">—</span></h2>
  <div class="book" id="book1">нет данных</div>
  <div class="kv" id="l1kv" style="margin-top:8px"></div>
 </div>
 <div class="panel">
  <h2>Стакан <span id="leg2name">—</span></h2>
  <div class="book" id="book2">нет данных</div>
  <div class="kv" id="l2kv" style="margin-top:8px"></div>
 </div>
 <div class="panel">
  <h2>Спред и сигнал</h2>
  <div class="row"><span>Z-score</span><span class="zbig" id="z">—</span></div>
  <div class="gauge" id="gauge">
    <div class="zone" id="zoneEntry"></div>
    <div class="mark" id="mark"></div>
  </div>
  <div class="row note"><span id="levels">—</span></div>
  <div class="kv" id="metrics" style="margin-top:10px"></div>
  <div class="note">Полный график входа — ниже, во всю ширину.</div>
 </div>
 <div class="panel" style="grid-column:1 / -1">
  <h2>График входа: Z-score, зоны входа/выхода/стопа и моменты, когда бот должен войти</h2>
  <svg id="chart" viewBox="0 0 1200 300" preserveAspectRatio="none"></svg>
  <div class="note" id="chartnote">—</div>
 </div>
 <div class="panel" style="grid-column:span 2">
  <h2>Условия входа сейчас</h2>
  <div class="row" style="align-items:baseline"><span>Готовность</span>
    <span><b id="entryDist" style="font-size:18px">—</b> <span class="note" id="entrySide"></span></span></div>
  <div class="gauge" id="progressGauge"><div class="mark" id="progressMark"></div></div>
  <div class="note" id="entryMissing">—</div>
  <div class="kv" id="entryChecks" style="margin-top:8px"></div>
 </div>
 <div class="panel">
  <h2>Бот</h2>
  <div class="kv" id="botkv"></div>
  <div id="positionBox" class="note"></div>
  <div class="kv" id="counters" style="margin-top:8px"></div>
 </div>
 <div class="panel">
  <h2>Сделки бота</h2>
  <table><thead><tr><th class="l">#</th><th class="l">вход</th><th class="l">выход</th>
  <th>сторона</th><th>цена вх.</th><th>цена вых.</th><th class="l">причина</th><th>мин</th><th>net</th></tr></thead>
  <tbody id="trades"><tr><td colspan="9" class="l">пока нет сделок</td></tr></tbody></table>
  <div class="note">Окно входа: PrimeHoursOnly (10:05–12:30, 15:30–18:35 MSK). Сигнал требует |Z| ≥ Z_entry.</div>
 </div>
 <div class="panel">
  <h2>Лента сделок (принты) <span id="printCounts" class="note"></span></h2>
  <table><thead><tr><th class="l">время</th><th class="l">тикер</th><th>цена</th><th>кол-во</th><th class="l">сторона</th></tr></thead>
  <tbody id="prints"><tr><td colspan="5" class="l">ожидание принтов…</td></tr></tbody></table>
 </div>
</div>
<script>
const fmt=(v,d=2)=>v===null||v===undefined?'—':Number(v).toFixed(d);
const sign=v=>v===null||v===undefined?'':(v>0?'pos':v<0?'neg':'');
async function tick(){
 let s;
 try{ s=await (await fetch('/api/state',{cache:'no-store'})).json(); }catch(e){ return; }
 document.getElementById('preset').textContent=`${s.preset} · ${s.leg1}/${s.leg2} · ${s.topology} · ${s.tariff}`;
 const b=document.getElementById('stateBadge');
 b.textContent=s.bot?`FSM: ${s.bot.state}`:'—'; b.className='badge state'+(s.bot&&s.bot.state!=='Idle'?'':'');
 document.getElementById('mode').textContent=s.params?`Zвх ${s.params.zEntry} · Zвых ${s.params.zExit} · стоп ${s.params.zStop} · сэмпл ${s.params.sampleSec}с`:'';
 document.getElementById('clock').textContent='MSK '+(s.serverTimeMsk||'—');
 document.getElementById('stats').textContent=s.runtime?`снимков ${s.runtime.snapshots} · ${s.runtime.seconds} c · ошибок ${s.runtime.restErrors}`:'';
 document.getElementById('leg1name').textContent='('+(s.leg1||'')+')';
 document.getElementById('leg2name').textContent='('+(s.leg2||'')+')';
 for(const leg of [1,2]){
  const book=s.books&&s.books['leg'+leg]; const el=document.getElementById('book'+leg);
  if(!book){ el.textContent='нет данных'; continue; }
  const maxQ=Math.max(1,...book.bids.map(x=>x.q),...book.asks.map(x=>x.q));
  let html='';
  [...book.asks].reverse().forEach((lv,i,arr)=>{
    const best=i===arr.length-1;
    html+=`<div class="lvl ask ${best?'best':''}"><div class="bar" style="width:${Math.round(lv.q/maxQ*46)}%"></div>
      <span>${fmt(lv.p,3)}</span><span></span><span class="q">${lv.q}</span></div>`;});
  html+=`<div class="midline"><span>спред</span><span>${fmt(book.bestAsk-book.bestBid,3)}</span></div>`;
  book.bids.forEach((lv,i)=>{
    const best=i===0;
    html+=`<div class="lvl bid ${best?'best':''}"><div class="bar" style="width:${Math.round(lv.q/maxQ*46)}%"></div>
      <span>${fmt(lv.p,3)}</span><span></span><span class="q">${lv.q}</span></div>`;});
  el.innerHTML=html;
  const kv=document.getElementById('l'+leg+'kv');
  const obi=leg===1?(s.metrics?s.metrics.obi1:null):(s.metrics?s.metrics.obi2:null);
  const micro=leg===1?(s.metrics?s.metrics.micro1:null):(s.metrics?s.metrics.micro2:null);
  kv.innerHTML=`<div><span>bid</span><b>${fmt(book.bestBid,3)}</b></div>
   <div><span>ask</span><b>${fmt(book.bestAsk,3)}</b></div>
   <div><span>micro</span><b>${fmt(micro,4)}</b></div>
   <div><span>OBI</span><b class="${sign(obi)}">${fmt(obi,3)}</b></div>`;
 }
 const m=s.metrics||{};
 document.getElementById('z').textContent=fmt(m.z,2);
 const zMax=Math.max(Math.abs(m.z||0),1)*1.15||1;
 const pos=(x)=>Math.max(0,Math.min(100,50+x/(s.params?s.params.zStop:6)*50));
 const mk=document.getElementById('mark'); if(mk) mk.style.left=pos(m.z||0)+'%';
 const zEntry=s.params?s.params.zEntry:2.8;
 const zone=document.getElementById('zoneEntry');
 if(zone){ zone.style.left=pos(-zEntry)+'%'; zone.style.width=(pos(zEntry)-pos(-zEntry))+'%'; }
 document.getElementById('levels').textContent=`шкала ±${(s.params?s.params.zStop:6)} · вход ±${zEntry}`;
 document.getElementById('metrics').innerHTML=`
  <div><span>спред</span><b>${fmt(m.spread,4)}</b></div>
  <div><span>μ</span><b>${fmt(m.mean,4)}</b></div>
  <div><span>σ</span><b>${fmt(m.sigma,4)}</b></div>
  <div><span>σ Leg1</span><b>${fmt(m.sigmaLeg1,4)}</b></div>
  <div><span>β slow</span><b>${fmt(m.beta,5)}</b></div>
  <div><span>β macro</span><b>${fmt(m.betaMacro,5)}</b></div>
  <div><span>дрейф β</span><b>${m.betaDriftPct===null?'—':fmt(m.betaDriftPct,3)+'%'}</b></div>
  <div><span>тиков</span><b>${m.ticks??'—'}${m.isWarm?'':' (прогрев)'}</b></div>
  <div><span>за день макс |Z|</span><b>${(s.day&&s.day.maxAbs!==undefined)?s.day.maxAbs.toFixed(2)+' (в '+(s.day.maxAt||'—')+')':'—'}</b></div>
  <div><span>за день выше порога</span><b>${(s.day&&s.day.aboveThreshold!==undefined)?s.day.aboveThreshold+' раз из '+s.day.samples:'—'}</b></div>
  <div><span>за день готовность</span><b>${(s.day&&s.day.readySamples!==undefined)?s.day.readySamples+' замеров':'—'}</b></div>`;
 const hist=(s.history||[]);
 if(hist.length>1){
  const W=1200,H=300,PAD=34;
  const zs=hist.map(p=>p.z);
  const zStop=s.params?s.params.zStop:6, zE=s.params?s.params.zEntry:2.8, zX=s.params?s.params.zExit:0.2;
  const zAbs=Math.max(...zs.map(Math.abs), zE*1.18, 0.6);
  const lo=-zAbs, hi=zAbs;
  const span=(hi-lo)||1;
  const X=i=>PAD+(i/(hist.length-1))*(W-2*PAD);
  const Y=v=>PAD+(1-(v-lo)/span)*(H-2*PAD);
  let svg='';
  // зоны входа: |Z| >= z_entry (две полосы)
  svg+=`<rect x="${PAD}" y="${Y(hi)}" width="${W-2*PAD}" height="${Math.max(Y(zE)-Y(hi),0)}" fill="rgba(79,163,255,.10)"/>`;
  svg+=`<rect x="${PAD}" y="${Y(-zE)}" width="${W-2*PAD}" height="${Math.max(Y(lo)-Y(-zE),0)}" fill="rgba(79,163,255,.10)"/>`;
  // линии уровней
  const line=(v,color,dash,label)=>`<line x1="${PAD}" y1="${Y(v)}" x2="${W-PAD}" y2="${Y(v)}" stroke="${color}" stroke-width="1" stroke-dasharray="${dash}" opacity=".85"/>
    <text x="${W-PAD+4}" y="${Y(v)+3}" fill="${color}" font-size="10">${label} ${v}</text>`;
  svg+=line(zE,'#4fa3ff','5 4','вход +'); svg+=line(-zE,'#4fa3ff','5 4','вход −');
  svg+=line(zX,'#7d8fa3','2 4','выход +'); svg+=line(-zX,'#7d8fa3','2 4','выход −');
  if(zStop<=zAbs){ svg+=line(zStop,'#ff5b6e','1 4','стоп +'); svg+=line(-zStop,'#ff5b6e','1 4','стоп −'); }
  // сетка по 1σ
  for(let g=Math.ceil(lo); g<=Math.floor(hi); g++){
    if(g===0) continue;
    svg+=`<line x1="${PAD}" y1="${Y(g)}" x2="${W-PAD}" y2="${Y(g)}" stroke="#1c2836" stroke-width="1"/>`;
    svg+=`<text x="6" y="${Y(g)+3}" fill="#4d5d70" font-size="9">${g>0?'+':''}${g}</text>`;
  }
  svg+=`<line x1="${PAD}" y1="${Y(0)}" x2="${W-PAD}" y2="${Y(0)}" stroke="#2c3b4d" stroke-width="1"/>`;
  // сам Z
  const pts=hist.map((p,i)=>`${X(i).toFixed(1)},${Y(p.z).toFixed(1)}`).join(' ');
  svg+=`<polyline points="${pts}" fill="none" stroke="#c678dd" stroke-width="1.8"/>`;
  // маркеры: где бот должен войти (все условия выполнены)
  hist.forEach((p,i)=>{ if(p.ready>=1){ svg+=`<circle cx="${X(i).toFixed(1)}" cy="${Y(p.z).toFixed(1)}" r="4.5" fill="#26d07c" stroke="#0b1016" stroke-width="1.2"/>`; } });
  // маркер последней точки
  const last=hist[hist.length-1];
  svg+=`<circle cx="${X(hist.length-1).toFixed(1)}" cy="${Y(last.z).toFixed(1)}" r="5" fill="#4fa3ff" stroke="#0b1016" stroke-width="1.5"/>`;
  svg+=`<text x="${PAD}" y="${H-8}" fill="#7d8fa3" font-size="10">точек ${hist.length} · Z от ${lo.toFixed(2)} до ${hi.toFixed(2)} · зелёные точки = все условия входа выполнены</text>`;
  document.getElementById('chart').innerHTML=svg;
  const readyCount=hist.filter(p=>p.ready>=1).length;
  const d=s.day||{};
  document.getElementById('chartnote').textContent=`В окне графика готовых входов: ${readyCount}. Сейчас Z=${last.z.toFixed(2)}. За весь день: макс |Z| ${d.maxAbs!==undefined?d.maxAbs.toFixed(2):'—'} (${d.maxAt||'—'}), выше порога ${d.aboveThreshold??'—'} замеров из ${d.samples??'—'}`;
 }
 const e=s.entry||{};
 document.getElementById('entryDist').textContent=e.distance===undefined?'—':(e.ready?'ВХОД РАЗРЕШЁН':`до входа ${e.distance.toFixed(2)}σ`);
 document.getElementById('entryDist').className=e.ready?'pos':'';
 document.getElementById('entrySide').textContent=e.side==='buy'?'(лонг Leg1)':'(шорт Leg1)';
 const pm=document.getElementById('progressMark');
 if(pm&&e.progressPct!==undefined){ pm.style.left=Math.max(0,Math.min(100,e.progressPct/100*100))+'%'; }
 document.getElementById('entryMissing').textContent=e.ready
   ? 'Все фильтры пройдены — бот выставит лимитную заявку в текущем тике.'
   : 'Мешает: '+((e.missing||[]).join('; ')||'—');
 const labels={z:'|Z| ≥ порога',hook:'хук пика Z',regime:'режимный фильтр',obi:'поток стакана (OBI)',
   vol:'σ / комиссия',edge:'возврат против комиссии',window:'окно входа',daily:'дневной предохранитель'};
 document.getElementById('entryChecks').innerHTML=Object.entries(e.checks||{}).map(([k,v])=>
   `<div><span>${labels[k]||k}</span><b class="${v?'pos':'neg'}">${v?'✓':'✗'}</b></div>
    <div style="grid-column:1/-1" class="note">${(e.details||{})[k]||''}</div>`).join('');
 const bot=s.bot||{};
 document.getElementById('botkv').innerHTML=`
  <div><span>состояние</span><b>${bot.state??'—'}</b></div>
  <div><span>сделок</span><b>${bot.trades??0}</b></div>
  <div><span>net</span><b class="${sign(bot.netRub)}">${fmt(bot.netRub,3)} RUB</b></div>
  <div><span>комиссии</span><b>${fmt(bot.feesRub,3)} RUB</b></div>
  <div><span>winrate</span><b>${fmt(bot.winRate,1)}%</b></div>
  <div><span>PF</span><b>${fmt(bot.profitFactor,2)}</b></div>`;
 document.getElementById('positionBox').textContent=bot.position
  ? `В позиции: ${bot.position.side.toUpperCase()} @ ${fmt(bot.position.entry,3)} с ${bot.position.openedMsk}, Z входа ${fmt(bot.position.entryZ,2)}, нереализовано ${fmt(bot.position.unrealized,3)} RUB`
  : (bot.dailyLocked?('Дневной предохранитель: '+bot.dailyLocked):'Позиции нет — ждём |Z| ≥ '+zEntry);
 const c=bot.counters||{};
 document.getElementById('counters').innerHTML=`
  <div><span>сигналов</span><b>${c.signals??0}</b></div>
  <div><span>входов</span><b>${c.entries??0}</b></div>
  <div><span>хук Z</span><b>${c.hookSkips??0}</b></div>
  <div><span>режим</span><b>${c.regimeSkips??0}</b></div>
  <div><span>σ/Fee</span><b>${c.volFilterSkips??0}</b></div>
  <div><span>окна</span><b>${c.sessionSkips??0}</b></div>`;
 const rows=(bot.tradeList||[]).map(t=>`<tr><td class="l">${t.id}</td><td class="l">${t.openMsk}</td>
  <td class="l">${t.closeMsk}</td><td>${t.side}</td><td>${fmt(t.entry,3)}</td><td>${fmt(t.exit,3)}</td>
  <td class="l">${t.reason}</td><td>${t.holdMin}</td><td class="${sign(t.net)}">${fmt(t.net,3)}</td></tr>`).join('');
 document.getElementById('trades').innerHTML=rows||'<tr><td colspan="9" class="l">пока нет сделок</td></tr>';
 const pr=(s.prints||[]).slice().reverse().map(p=>`<tr><td class="l">${p.timeMsk}</td><td class="l">${p.ticker}</td>
  <td>${fmt(p.price,3)}</td><td>${p.qty}</td><td class="l">${p.side}</td></tr>`).join('');
 document.getElementById('prints').innerHTML=pr||'<tr><td colspan="5" class="l">ожидание принтов…</td></tr>';
 const pc=s.printCounts||{}; document.getElementById('printCounts').textContent=Object.entries(pc).map(([k,v])=>`${k}: ${v}`).join(' · ');
}
setInterval(tick,400); tick();
</script></body></html>
"""
