"""Веб-дашборд свинг-корзины: графики по каждому фьючерсу, позиции, портфель.

Без внешних зависимостей: `http.server` из stdlib + SVG-графики на клиенте.
Сервер поднимается в фоновом потоке рядом с торговым циклом и читает живой
снапшот из `SwingBasketRunner.snapshot()`.

    from bot_a.services.swing_dashboard import SwingDashboard
    dashboard = SwingDashboard(runner, port=8099)
    dashboard.start()   # http://127.0.0.1:8099/
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional


class _Handler(BaseHTTPRequestHandler):
    runner: Any = None

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 — тишина в логах
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:  # noqa: N802 — API стандартной библиотеки
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/snapshot":
            try:
                payload = self.runner.snapshot() if self.runner is not None else {}
            except Exception as exc:  # noqa: BLE001 — дашборд не должен падать
                payload = {"error": str(exc)}
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if path == "/health":
            self._send(200, b"ok", "text/plain; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")


class SwingDashboard:
    """HTTP-сервер дашборда в отдельном потоке."""

    def __init__(self, runner: Any, host: str = "127.0.0.1", port: int = 8099) -> None:
        self._runner = runner
        self._host = host
        self._port = int(port)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        if self._server is not None:
            return int(self._server.server_address[1])
        return self._port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}/"

    def start(self) -> "SwingDashboard":
        if self._server is not None:
            return self
        handler = type("_BoundHandler", (_Handler,), {"runner": self._runner})
        server = ThreadingHTTPServer((self._host, self._port), handler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="swing-dashboard", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None


PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Bot A — свинг-корзина FORTS</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 16px 20px 40px; background: #0e1116; color: #d7dde5;
         font: 14px/1.45 "Segoe UI", Roboto, sans-serif; }
  h1 { font-size: 17px; margin: 0 0 4px; font-weight: 600; }
  .muted { color: #7d8794; font-size: 12px; }
  .tiles { display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0 18px; }
  .tile { background: #171b22; border: 1px solid #232a34; border-radius: 8px;
          padding: 8px 12px; min-width: 132px; }
  .tile .k { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #7d8794; }
  .tile .v { font-size: 18px; font-weight: 600; margin-top: 2px; }
  .pos { color: #35c07a; } .neg { color: #ef5b5b; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(430px, 1fr)); gap: 14px; }
  .card { background: #131820; border: 1px solid #232a34; border-radius: 10px; padding: 10px 12px 12px; }
  .card h2 { font-size: 14px; margin: 0; font-weight: 600; display: flex; justify-content: space-between; }
  .card h2 .badge { font-size: 11px; padding: 1px 7px; border-radius: 999px; border: 1px solid #2d6e4f;
                    background: #123024; color: #4fd08a; }
  .card h2 .badge.short { border-color: #7a3030; background: #2a1414; color: #f07d7d; }
  .card h2 .badge.flat { border-color: #2a3340; background: #161c24; color: #7d8794; }
  .kv { display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; color: #9aa5b1; margin: 6px 0 8px; }
  .kv b { color: #d7dde5; font-weight: 600; }
  table { border-collapse: collapse; width: 100%; font-size: 12px; }
  th, td { text-align: right; padding: 3px 6px; border-bottom: 1px solid #1d242e; white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  th { color: #7d8794; font-weight: 500; }
  .legend { font-size: 11px; color: #7d8794; margin-top: 4px; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
</style>
</head>
<body>
<h1>Bot A — свинг-корзина FORTS</h1>
<div class="muted" id="sub">загрузка…</div>
<div class="tiles" id="tiles"></div>
<div class="grid" id="cards"></div>
<h2 style="font-size:14px;margin:22px 0 6px">Последние сделки</h2>
<div style="overflow:auto"><table id="trades"></table></div>
<div class="legend">▲ вход Long · ▼ вход Short · ✕ выход · <span class="dot" style="background:#ef5b5b"></span>стоп ·
<span class="dot" style="background:#4fd08a"></span>цель (EMA20 для фейдов)</div>
<script>
const fmt = (v, d = 2) => (v === null || v === undefined) ? '—' : Number(v).toLocaleString('ru-RU', {minimumFractionDigits: d, maximumFractionDigits: d});
const cls = v => (v > 0 ? 'pos' : (v < 0 ? 'neg' : ''));
const time = ms => new Date(ms).toLocaleString('ru-RU', {day:'2-digit', month:'2-digit', hour:'2-digit', minute:'2-digit'});

function svgChart(inst) {
  const W = 420, H = 170, P = 6;
  const bars = inst.bars || [];
  const pos = inst.position;
  if (!bars.length) return '<div class="muted">нет баров для графика</div>';
  const values = bars.map(b => b[4]);
  if (pos) { values.push(pos.stop); if (pos.target) values.push(pos.target); }
  let lo = Math.min(...values), hi = Math.max(...values);
  if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
  const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  const x = i => P + (W - 2 * P) * (bars.length === 1 ? 0.5 : i / (bars.length - 1));
  const y = v => H - P - (H - 2 * P) * (v - lo) / (hi - lo);
  const line = bars.map((b, i) => (i ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(b[4]).toFixed(1)).join(' ');
  let out = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" preserveAspectRatio="none">`;
  out += `<rect x="0" y="0" width="${W}" height="${H}" fill="#0f141b" rx="6"/>`;
  out += `<path d="${line}" fill="none" stroke="#4f9dff" stroke-width="1.6"/>`;
  const t0 = bars[0][0], t1 = bars[bars.length - 1][0];
  const tx = ms => P + (W - 2 * P) * (t1 === t0 ? 0.5 : (ms - t0) / (t1 - t0));
  (inst.trades || []).forEach(t => {
    const cx = tx(t.entryMs), cy = y(t.entry);
    const up = t.side === 'Long';
    out += `<path d="M${cx} ${cy} l5 ${up ? 8 : -8} l-10 0 z" fill="${up ? '#35c07a' : '#ef5b5b'}"/>`;
    const ex = tx(t.exitMs), ey = y(t.exit);
    out += `<path d="M${ex - 3} ${ey - 3} l6 6 M${ex + 3} ${ey - 3} l-6 6" stroke="${t.netRub >= 0 ? '#35c07a' : '#ef5b5b'}" stroke-width="1.4"/>`;
  });
  if (pos) {
    out += `<line x1="${P}" x2="${W - P}" y1="${y(pos.stop)}" y2="${y(pos.stop)}" stroke="#ef5b5b" stroke-dasharray="4 3" stroke-width="1.2"/>`;
    if (pos.target) out += `<line x1="${P}" x2="${W - P}" y1="${y(pos.target)}" y2="${y(pos.target)}" stroke="#4fd08a" stroke-dasharray="4 3" stroke-width="1.2"/>`;
    const ex = tx(pos.entryMs), ey = y(pos.entry);
    out += `<circle cx="${ex}" cy="${ey}" r="3.4" fill="none" stroke="#ffd166" stroke-width="1.4"/>`;
  }
  out += `<text x="${P + 2}" y="12" fill="#6f7b88" font-size="10">${fmt(hi, 2)}</text>`;
  out += `<text x="${P + 2}" y="${H - 2}" fill="#6f7b88" font-size="10">${fmt(lo, 2)}</text>`;
  out += '</svg>';
  return out;
}

function card(inst) {
  const pos = inst.position;
  const badge = !pos ? '<span class="badge flat">вне рынка</span>'
    : `<span class="badge ${pos.side === 'Short' ? 'short' : ''}">${pos.side} ${pos.qty} @ ${fmt(pos.entry, 2)}</span>`;
  let kv = `<div class="kv"><span>last <b>${fmt(inst.last, 2)}</b></span>
            <span>ATR <b>${fmt(inst.atr, 2)}</b></span><span>ER20 <b>${fmt(inst.er, 2)}</b></span>
            <span>EMA20 <b>${fmt(inst.emaFast, 2)}</b></span><span>EMA90 <b>${fmt(inst.emaSlow, 2)}</b></span>`;
  if (inst.emaHtf) kv += `<span>EMA_HTF <b>${fmt(inst.emaHtf, 2)}</b></span>`;
  kv += '</div>';
  if (pos) {
    kv += `<div class="kv"><span>стоп <b class="neg">${fmt(pos.stop, 2)}</b> (${fmt(pos.stopDistancePoints, 2)} пт)</span>
      ${pos.target ? `<span>цель <b class="pos">${fmt(pos.target, 2)}</b></span>` : ''}
      <span>PnL <b class="${cls(pos.unrealizedRub)}">${fmt(pos.unrealizedRub, 2)} ₽</b></span>
      <span>MFE <b>${fmt(pos.mfe, 2)}</b></span><span>держим <b>${fmt(pos.holdMin, 0)} мин</b></span>
      <span>${pos.breakevenDone ? 'стоп в прибыли' : 'стоп под риском'}</span>
      ${pos.trailActive ? '<span>трейл вкл.</span>' : ''}</div>`;
  }
  const rows = (inst.trades || []).slice(-6).reverse().map(t =>
    `<tr><td>${time(t.entryMs)}</td><td>${t.side === 'Long' ? 'L' : 'S'} ${t.qty}</td>
     <td>${fmt(t.entry, 2)} → ${fmt(t.exit, 2)}</td><td>${t.reason}</td>
     <td class="${cls(t.netRub)}">${fmt(t.netRub, 2)}</td></tr>`).join('');
  return `<div class="card"><h2>${inst.ticker} ${badge}</h2>${kv}${svgChart(inst)}
    <table><thead><tr><th>вход</th><th>поз.</th><th>цены</th><th>причина</th><th>нетто ₽</th></tr></thead>
    <tbody>${rows || '<tr><td colspan="5" class="muted">сделок ещё не было</td></tr>'}</tbody></table></div>`;
}

function render(data) {
  const p = data.portfolio || {};
  document.getElementById('sub').textContent =
    `${data.mode || ''} · бары ${data.barMinutes} мин · обновлено ${new Date(data.generatedMs).toLocaleTimeString('ru-RU')} · риск ${p.riskPct}% на сделку`;
  const tiles = [
    ['Эквити', `${fmt(p.equity, 0)} ₽`, cls((p.equity || 0) - (p.capital || 0))],
    ['Реализовано', `${fmt(p.realizedRub, 0)} ₽`, cls(p.realizedRub)],
    ['Нереализовано', `${fmt(p.unrealizedRub, 0)} ₽`, cls(p.unrealizedRub)],
    ['Комиссии', `${fmt(p.feesRub, 0)} ₽`, ''],
    ['ГО занято', `${fmt(p.marginUsed, 0)} / ${fmt(p.marginLimitRub, 0)} ₽`, ''],
    ['Позиции', `${p.openPositions || 0} (риск-слотов ${p.maxRiskPositions || 0})`, ''],
    ['Сделок', `${p.trades || 0}`, ''],
    ['Win rate', `${fmt(p.winRate, 1)} %`, ''],
    ['Profit factor', p.profitFactor === null ? '—' : fmt(p.profitFactor, 2), ''],
  ];
  document.getElementById('tiles').innerHTML = tiles.map(([k, v, c]) =>
    `<div class="tile"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join('');
  const insts = data.instruments || [];
  document.getElementById('cards').innerHTML = insts.map(card).join('');
  const all = [];
  insts.forEach(i => (i.trades || []).forEach(t => all.push({...t, ticker: i.ticker})));
  all.sort((a, b) => b.exitMs - a.exitMs);
  const rows = all.slice(0, 40).map(t =>
    `<tr><td>${t.ticker}</td><td>${time(t.entryMs)} → ${time(t.exitMs)}</td>
     <td>${t.side === 'Long' ? 'L' : 'S'} ${t.qty}</td><td>${fmt(t.entry, 2)} → ${fmt(t.exit, 2)}</td>
     <td>${fmt(t.holdMin, 0)} мин</td><td>${t.reason}</td><td>${fmt(t.feesRub, 2)}</td>
     <td class="${cls(t.netRub)}">${fmt(t.netRub, 2)}</td></tr>`).join('');
  document.getElementById('trades').innerHTML =
    `<thead><tr><th>тикер</th><th>вход → выход</th><th>поз.</th><th>цены</th><th>держали</th>
     <th>причина</th><th>комиссия ₽</th><th>нетто ₽</th></tr></thead>
     <tbody>${rows || '<tr><td colspan="8" class="muted">сделок ещё не было</td></tr>'}</tbody>`;
}

async function tick() {
  try {
    const res = await fetch('/api/snapshot', {cache: 'no-store'});
    render(await res.json());
  } catch (e) {
    document.getElementById('sub').textContent = 'нет связи с ботом: ' + e;
  }
}
tick();
setInterval(tick, 3000);
</script>
</body>
</html>
"""
