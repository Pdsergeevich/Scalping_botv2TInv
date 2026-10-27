"""Мульти-парный раннер `--preset ALL_ACTIVE` (v2.5).

Один процесс ведёт все включённые пары одновременно: у каждой свой экземпляр ядра,
свой дневной предохранитель, свой `PairBookSyncer` и свой `_DayTracker`. Данные идут
из того же REST-поллинга T-Invest, страница показывает все пары рядом.

Режимы:
* ``watch`` — только наблюдение (исполнение симулятором, деньги не тратятся);
* ``paper`` — тот же путь, но с журналом сделок и периодической сводкой в консоль.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence

from bot_a.broker.tinkoff_rest_client import TInvestRestClient, TInvestRestError
from bot_a.config import AppConfig, PairPreset
from bot_a.economics.fees import TinkoffFeeCalculator
from bot_a.engine.factory import create_engine, create_guard
from bot_a.engine.ledger import PortfolioLedger
from bot_a.execution.simulated_gateway import SimulatedOrderExecutionGateway
from bot_a.services.l2_sync import PairBookSyncer
from bot_a.services.live_dashboard import LiveWatchService, _DayTracker, _f
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import MSK, RealTimeProvider, ms_to_msk

LOGGER_NAME = "bot_a.multi"
DEFAULT_PORT = 8099


@dataclass(slots=True)
class PairRuntime:
    """Всё, что нужно одной паре для независимой работы внутри общего процесса."""

    preset: PairPreset
    engine: Any
    ledger: PortfolioLedger
    gateway: Any
    syncer: PairBookSyncer
    day: Any
    books: Dict[int, Any] = field(default_factory=dict)
    prints: List[Dict[str, Any]] = field(default_factory=list)
    seen_trades: set = field(default_factory=set)
    rest_errors: int = 0
    snapshots: int = 0
    last_seen_trade_id: int = 0

    @property
    def name(self) -> str:
        return self.preset.name


class _MultiState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._payload: Dict[str, Any] = {"status": "starting", "pairs": []}

    def publish(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._payload = payload

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._payload)


class _MultiHandler(BaseHTTPRequestHandler):
    state: _MultiState = _MultiState()
    page: bytes = b""

    def do_GET(self) -> None:  # noqa: N802 — интерфейс BaseHTTPRequestHandler
        if self.path.startswith("/api/state"):
            body = json.dumps(self.state.snapshot(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = self.page
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — как в базовом классе
        return


class MultiPresetRunner:
    """Ведёт несколько пар в одном процессе и отдаёт общую страницу."""

    def __init__(
        self,
        config: AppConfig,
        presets: Sequence[PairPreset],
        client: TInvestRestClient,
        interval_ms: int = 300,
        depth: int = 20,
        port: int = DEFAULT_PORT,
        serve_dashboard: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._presets = list(presets)
        self._client = client
        self._interval_ms = max(int(interval_ms), 100)
        self._depth = max(int(depth), 5)
        self._port = int(port)
        self._serve_dashboard = bool(serve_dashboard)
        self._log = logger or logging.getLogger(LOGGER_NAME)
        self._state = _MultiState()
        self._runtimes: List[PairRuntime] = []
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ сборка
    async def _build_runtime(self, preset: PairPreset) -> Optional[PairRuntime]:
        clock = RealTimeProvider()
        legs = [preset.leg1, preset.leg2]
        fee = TinkoffFeeCalculator(self._config.fees.tariff, clock=clock)
        gateway = SimulatedOrderExecutionGateway(
            metadata=legs,
            clock=clock,
            fee_calculator=fee,
            latency_ms=self._config.execution.simulated_order_latency_ms,
            queue_fill_factor=self._config.execution.queue_fill_factor,
            slippage_steps=self._config.execution.slippage_steps,
            order_book_depth=self._depth,
            initial_cash_rub=self._config.execution.paper_initial_cash_rub,
        )
        guard = create_guard(self._config, legs)
        ledger = PortfolioLedger(self._config.execution.paper_initial_cash_rub)
        params = self._params_for(preset)
        engine = create_engine(
            legs,
            params,
            self._topology_for(preset),
            gateway,
            fee,
            clock,
            guard,
            ledger,
            self._log,
            daily_state_path=str(
                self._config.path(self._config.storage.state_dir)
                / f"daily_state_{preset.name}.json"
            ),
        )
        await gateway.start()
        runtime = PairRuntime(
            preset=preset,
            engine=engine,
            ledger=ledger,
            gateway=gateway,
            syncer=PairBookSyncer(
                interval_ms=self._interval_ms,
                skew_factor=float(self._config.execution.l2_skew_factor),
            ),
            day=_DayTracker(preset.name, self._config.path(self._config.storage.state_dir)),
        )
        days = int(self._config.execution.bootstrap_warmup_days)
        if days > 0:
            from bot_a.services.bootstrap import bootstrap_warmup

            try:
                await bootstrap_warmup(self._config, preset, engine, days=days, logger=self._log)
            except Exception as exc:  # noqa: BLE001 — прогрев не должен ронять пару
                self._log.warning("%s: прогрев не удался (%s): %s", preset.name, type(exc).__name__, exc)
                engine.set_bootstrap(False)
        return runtime

    def _params_for(self, preset: PairPreset) -> Any:
        from dataclasses import replace

        base = self._config.strategy
        return replace(
            base,
            beta=float(preset.beta),
            beta_mode=(preset.beta_mode if base.beta_mode in ("", "auto") else preset.beta_mode),
            stat_sample_interval_seconds=float(self._config.execution.stat_sample_interval_seconds),
            use_entry_persistence=bool(self._config.execution.l2_entry_persistence),
            entry_confirm_samples=int(self._config.execution.l2_entry_confirm_samples),
            z_entry=float(preset.z_entry) if preset.z_entry is not None else base.z_entry,
            ewma_window=int(preset.ewma_window) if preset.ewma_window is not None else base.ewma_window,
            min_sigma_to_fee_ratio=(
                float(preset.min_sigma_to_fee_ratio)
                if preset.min_sigma_to_fee_ratio is not None
                else base.min_sigma_to_fee_ratio
            ),
            tp_deviation_fraction=(
                float(preset.tp_deviation_fraction)
                if preset.tp_deviation_fraction is not None
                else base.tp_deviation_fraction
            ),
            tp_min_fee_multiple=(
                dec(preset.tp_min_fee_multiple)
                if preset.tp_min_fee_multiple is not None
                else base.tp_min_fee_multiple
            ),
            quantity_lots=(
                max(int(preset.quantity_lots), 1)
                if preset.quantity_lots is not None
                else base.quantity_lots
            ),
        )

    def _topology_for(self, preset: PairPreset) -> Any:
        from bot_a.models.domain import ExecutionTopology

        if preset.topology:
            return ExecutionTopology(preset.topology)
        return self._config.execution.topology

    # ------------------------------------------------------------------ цикл
    async def run(
        self,
        duration_seconds: Optional[float] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        for preset in self._presets:
            try:
                leg1 = await self._client.resolve_instrument(
                    preset.leg1.ticker, preset.leg1.class_code, preset.leg1.market
                )
                leg2 = await self._client.resolve_instrument(
                    preset.leg2.ticker, preset.leg2.class_code, preset.leg2.market
                )
            except TInvestRestError as exc:
                self._log.error("%s: инструменты не разрешены (%s)", preset.name, exc)
                continue
            resolved = preset.with_metadata(leg1, leg2)
            runtime = await self._build_runtime(resolved)
            if runtime is not None:
                self._runtimes.append(runtime)
                self._log.info(
                    "%s подключена: %s uid=%s, %s uid=%s",
                    resolved.name,
                    leg1.ticker,
                    leg1.instrument_uid,
                    leg2.ticker,
                    leg2.instrument_uid,
                )
        if not self._runtimes:
            self._log.error("ни одна пара не поднялась — проверьте пресеты и токен")
            return {"pairs": []}

        if self._serve_dashboard:
            self._start_server()
        started = time.monotonic()
        deadline = (started + duration_seconds) if duration_seconds else None
        last_trades_poll = 0.0
        last_console = 0.0
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    self._log.info("Длительность сессии истекла")
                    break
                for runtime in self._runtimes:
                    await self._poll_pair(runtime)
                now = time.monotonic()
                if now - last_trades_poll >= 1.0:
                    last_trades_poll = now
                    for runtime in self._runtimes:
                        try:
                            await self._poll_trades(runtime)
                        except Exception as exc:  # noqa: BLE001
                            runtime.rest_errors += 1
                            self._log.warning("%s: GetLastTrades: %s", runtime.name, exc)
                if self._serve_dashboard:
                    self._publish(started)
                if now - last_console >= 30.0:
                    last_console = now
                    self._print_console(started)
                await asyncio.sleep(self._interval_ms / 1000.0)
        except KeyboardInterrupt:
            self._log.warning("Прервано пользователем")
        finally:
            for runtime in self._runtimes:
                try:
                    await runtime.engine.shutdown(flatten=self._config.risk.flatten_on_shutdown)
                except Exception:  # noqa: BLE001
                    pass
                await runtime.gateway.stop()
            self._stop_server()
        return {"pairs": [self._pair_payload(runtime) for runtime in self._runtimes]}

    async def _poll_pair(self, runtime: PairRuntime) -> None:
        fresh: Dict[int, Any] = {}
        fetched_at: Dict[int, float] = {}
        failed: List[int] = []
        for leg, meta in ((1, runtime.preset.leg1), (2, runtime.preset.leg2)):
            try:
                book = await self._client.get_order_book(meta.instrument_uid, self._depth)
            except (TInvestRestError, TimeoutError, ConnectionError, OSError) as exc:
                runtime.rest_errors += 1
                failed.append(leg)
                if runtime.rest_errors % 20 == 1:
                    self._log.warning("%s: GetOrderBook %s: %s", runtime.name, meta.ticker, exc)
                continue
            if book is None:
                failed.append(leg)
                continue
            runtime.books[leg] = book
            fresh[leg] = book
            fetched_at[leg] = time.monotonic() * 1000.0
        decision = runtime.syncer.check(fresh, fetched_at, failed)
        if decision.ok:
            for leg in (1, 2):
                await runtime.engine.on_order_book(leg, fresh[leg])
                runtime.snapshots += 1
        elif runtime.syncer.skew_skips % 50 == 1:
            self._log.warning(
                "%s: рассинхрон стаканов (%d): %s", runtime.name, runtime.syncer.skew_skips, decision.reason
            )
        await runtime.engine.on_clock()

    async def _poll_trades(self, runtime: PairRuntime) -> None:
        since = datetime.now(tz=timezone.utc) - timedelta(seconds=60)
        for leg, meta in ((1, runtime.preset.leg1), (2, runtime.preset.leg2)):
            try:
                trades = await self._client.get_last_trades(meta.instrument_uid, since)
            except TInvestRestError:
                continue
            for trade in trades:
                key = (meta.instrument_uid, trade.time_ms, str(trade.price), trade.quantity)
                if key in runtime.seen_trades:
                    continue
                runtime.seen_trades.add(key)
                await runtime.engine.on_trade(leg, trade)
                runtime.prints.append(
                    {
                        "ticker": meta.ticker,
                        "timeMsk": ms_to_msk(trade.time_ms).strftime("%H:%M:%S"),
                        "price": _f(trade.price),
                        "qty": trade.quantity,
                    }
                )
                del runtime.prints[:-200]

    # ------------------------------------------------------------------ вывод
    def _pair_payload(self, runtime: PairRuntime) -> Dict[str, Any]:
        engine = runtime.engine
        metrics = engine.metrics
        params = engine.params
        entry = (
            LiveWatchService._entry_diagnostics(None, engine, metrics)  # noqa: SLF001
            if metrics is not None
            else {"ready": False, "distance": params.z_entry, "checks": {}, "missing": ["нет метрик"]}
        )
        runtime.day.feed(metrics, bool(entry.get("ready")), float(params.z_entry))
        snapshot = engine.snapshot()
        fee = float(engine._current_round_trip_fee(params.quantity_lots))  # noqa: SLF001
        sigma = float(engine._sigma_leg1_rub(metrics)) if metrics is not None else 0.0  # noqa: SLF001
        return {
            "name": runtime.name,
            "leg1": runtime.preset.leg1.ticker,
            "leg2": runtime.preset.leg2.ticker,
            "topology": self._topology_for(runtime.preset).value,
            "zEntry": round(float(params.z_entry), 3),
            "zExit": round(float(params.z_exit), 3),
            "state": snapshot["state"],
            "z": None if metrics is None else round(float(metrics.zscore), 3),
            "distance": round(float(entry.get("distance", 0.0)), 3),
            "ready": bool(entry.get("ready")),
            "missing": entry.get("missing", []),
            "checks": entry.get("checks", {}),
            "sigmaLeg1": round(sigma, 5),
            "feeRub": round(fee, 5),
            "sigmaToFee": round(sigma / fee, 2) if fee else None,
            "betaDriftPct": None
            if metrics is None or metrics.beta_drift is None
            else round(float(metrics.beta_drift) * 100.0, 3),
            "obi1": None if metrics is None else round(float(metrics.obi_leg1), 3),
            "trades": snapshot["trades"],
            "netRub": _f(snapshot["netPnlRub"]),
            "feesRub": _f(snapshot["feesRub"]),
            "dailyLocked": snapshot.get("dailyLockedReason"),
            "counters": snapshot.get("counters", {}),
            "position": None
            if engine.position is None
            else {
                "side": engine.position.side_leg1.value,
                "entry": _f(engine.position.leg1_entry_price),
                "entryZ": round(float(engine.position.entry_z), 2),
                "openedMsk": ms_to_msk(engine.position.opened_ms).strftime("%H:%M:%S"),
            },
            "snapshots": runtime.snapshots,
            "restErrors": runtime.rest_errors,
            "skewSkips": runtime.syncer.skew_skips,
            "day": runtime.day.payload(float(params.z_entry)),
            "book": None
            if 1 not in runtime.books
            or not runtime.books[1].bids
            or not runtime.books[1].asks
            else {
                "bid": _f(runtime.books[1].bids[0].price),
                "ask": _f(runtime.books[1].asks[0].price),
            },
        }

    def _publish(self, started: float) -> None:
        self._state.publish(
            {
                "status": "live",
                "serverTimeMsk": datetime.now(tz=MSK).strftime("%H:%M:%S"),
                "seconds": round(time.monotonic() - started, 1),
                "intervalMs": self._interval_ms,
                "presets": len(self._runtimes),
                "pairs": [self._pair_payload(runtime) for runtime in self._runtimes],
            }
        )

    def _print_console(self, started: float) -> None:
        elapsed = time.monotonic() - started
        header = (
            f"{'пара':<14}{'Z':>7}{'до входа':>10}{'S/Fee':>7}{'βдрейф%':>9}"
            f"{'OBI':>7}{'сдел.':>7}{'net':>9}{'макс|Z|':>9}{'skew':>6}"
        )
        self._log.info("── сводка (%.0f c) ──\n%s", elapsed, header)
        for runtime in self._runtimes:
            payload = self._pair_payload(runtime)
            z = payload["z"]
            day = payload["day"]
            self._log.info(
                "%-14s%7s%10.2f%7s%9s%7s%7d%9.3f%9.2f%6d",
                payload["name"],
                "—" if z is None else f"{z:+.2f}",
                payload["distance"],
                "—" if payload["sigmaToFee"] is None else f"{payload['sigmaToFee']:.2f}",
                "—" if payload["betaDriftPct"] is None else f"{payload['betaDriftPct']:+.2f}",
                "—" if payload["obi1"] is None else f"{payload['obi1']:+.2f}",
                payload["trades"],
                payload["netRub"],
                day.get("maxAbs", 0.0),
                payload["skewSkips"],
            )

    # ------------------------------------------------------------------ HTTP
    def _start_server(self) -> str:
        _MultiHandler.state = self._state
        _MultiHandler.page = MULTI_HTML.encode("utf-8")
        self._server = ThreadingHTTPServer(("127.0.0.1", self._port), _MultiHandler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        url = f"http://127.0.0.1:{self._port}/"
        self._log.info("Мульти-дашборд доступен: %s", url)
        return url

    def _stop_server(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


MULTI_HTML = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>Bot A — портфель</title>
<style>
:root{--bg:#0b1016;--panel:#131b24;--line:#22303f;--text:#dbe6f2;--muted:#7d8fa3;
--up:#26d07c;--down:#ff5b6e;--accent:#4fa3ff;--warn:#f5a623;--z:#c678dd}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 'Segoe UI',system-ui,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:center}
h1{font-size:15px;margin:0;font-weight:600}
.badge{padding:3px 9px;border-radius:11px;background:#1b2734;border:1px solid var(--line);font-size:12px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:7px 8px;border-bottom:1px solid #1a2530;text-align:right;font-size:13px}
th{color:var(--muted);font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.4px}
td.name{text-align:left;font-weight:600}
tr.ready{background:rgba(38,208,124,.10)}
.pos{color:var(--up)}.neg{color:var(--down)}.muted{color:var(--muted)}
.wrap{padding:12px 16px}
.hint{margin-top:10px;color:var(--muted);font-size:12px}
.pairz{font-weight:700}
</style></head><body>
<header><h1>Bot A — портфель (ALL_ACTIVE)</h1>
<span class="badge" id="clock">—</span>
<span class="badge" id="mode">—</span></header>
<div class="wrap">
<table><thead><tr>
<th style="text-align:left">пара</th><th>состояние</th><th>Z</th><th>до входа</th><th>σ/Fee</th>
<th>β дрейф</th><th>OBI</th><th>сделок</th><th>net RUB</th><th>день: макс |Z|</th>
<th>skew</th><th>ошибки</th>
</tr></thead><tbody id="rows"><tr><td colspan="12" class="muted">загрузка…</td></tr></tbody></table>
<div class="hint">Зелёная строка — все 8 фильтров входа выполнены прямо сейчас. Данные: REST
MarketDataService/GetOrderBook обеих ног с проверкой синхронности (v2.5), исполнение — симулятор.</div>
</div>
<script>
const fmt=(v,d)=>v===null||v===undefined?'—':Number(v).toFixed(d);
const sign=v=>v===null||v===undefined?'':(v>0?'pos':(v<0?'neg':''));
async function tick(){
 try{
  const r=await fetch('/api/state',{cache:'no-store'});
  const s=await r.json();
  document.getElementById('clock').textContent='MSK '+s.serverTimeMsk+' · '+Math.round(s.seconds)+' c';
  document.getElementById('mode').textContent='пар: '+(s.pairs||[]).length+' · опрос '+s.intervalMs+' мс';
  const rows=(s.pairs||[]).map(p=>{
   const st=p.state==='Idle'?'Idle':p.state;
   return `<tr class="${p.ready?'ready':''}">
    <td class="name">${p.leg1}/${p.leg2}<div class="muted" style="font-weight:400;font-size:11px">${p.topology}</div></td>
    <td>${st}</td>
    <td class="pairz ${sign(p.z)}">${fmt(p.z,2)}</td>
    <td>${p.ready?'<b style="color:#26d07c">ГОТОВ</b>':fmt(p.distance,2)+'σ'}</td>
    <td class="${p.sigmaToFee!==null&&p.sigmaToFee<1.3?'neg':'pos'}">${fmt(p.sigmaToFee,2)}</td>
    <td class="${Math.abs(p.betaDriftPct||0)>0.8?'neg':'muted'}">${fmt(p.betaDriftPct,2)}%</td>
    <td class="${sign(p.obi1)}">${fmt(p.obi1,2)}</td>
    <td>${p.trades}</td>
    <td class="${sign(p.netRub)}">${fmt(p.netRub,3)}</td>
    <td>${fmt(p.day?p.day.maxAbs:null,2)}</td>
    <td>${p.skewSkips}</td>
    <td class="${p.restErrors?'neg':'muted'}">${p.restErrors}</td></tr>`;
  }).join('');
  document.getElementById('rows').innerHTML=rows||'<tr><td colspan="12" class="muted">нет пар</td></tr>';
 }catch(e){ document.getElementById('clock').textContent='нет связи'; }
}
tick(); setInterval(tick, 2000);
</script></body></html>
"""
