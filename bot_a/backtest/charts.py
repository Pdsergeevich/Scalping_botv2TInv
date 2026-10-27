"""Визуализация работы бота: графики входов/выходов, Z-score, эквити и разбор сделок.

Модуль не влияет на торговую логику: он получает уже посчитанные метрики
(``TraceRecorder``) и завершённые сделки (``TradeRecord``) и рисует PNG + HTML-страницу.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from bot_a.backtest.candles import Candle  # noqa: E402
from bot_a.models.domain import ExitReason, Side, SpreadMetrics, TradeRecord  # noqa: E402
from bot_a.utils.time_provider import MSK, ms_to_msk  # noqa: E402

LOGGER_NAME = "bot_a.charts"

COLORS = {
    "bg": "#0f1419",
    "panel": "#161d26",
    "grid": "#243040",
    "text": "#d7e0ea",
    "accent": "#4fa3ff",
    "leg1": "#4fa3ff",
    "leg2": "#f5a623",
    "buy": "#2ecc71",
    "sell": "#ff5c5c",
    "profit": "#2ecc71",
    "loss": "#ff5c5c",
    "neutral": "#9aa7b4",
    "zline": "#c678dd",
}

EXIT_COLORS = {
    ExitReason.TAKE_PROFIT.value: "#2ecc71",
    ExitReason.TIME_STOP.value: "#f5a623",
    ExitReason.HARD_STOP.value: "#ff5c5c",
    ExitReason.SCRATCH.value: "#9aa7b4",
    ExitReason.BREAKEVEN.value: "#4fa3ff",
    ExitReason.EMERGENCY.value: "#c678dd",
    ExitReason.SHUTDOWN.value: "#8b949e",
}

PRIME_WINDOWS = ((time(10, 5), time(12, 30)), (time(15, 30), time(18, 35)))


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["panel"],
            "savefig.facecolor": COLORS["bg"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.55,
            "font.size": 9,
            "legend.facecolor": COLORS["panel"],
            "legend.edgecolor": COLORS["grid"],
            "figure.autolayout": False,
        }
    )


def _as_naive(moment_ms: int) -> datetime:
    return datetime.fromtimestamp(moment_ms / 1000, tz=MSK).replace(tzinfo=None)


def _fmt_rub(value: Any) -> str:
    return f"{float(value):+.2f}"


@dataclass(slots=True)
class TraceRecorder:
    """Разреженная траектория метрик ядра для графиков.

    Записывает не чаще ``min_interval_ms`` (по умолчанию раз в минуту), плюс
    принудительно фиксирует точки на моментах входов и выходов.
    """

    min_interval_ms: int = 60_000
    points: List[Dict[str, float]] = field(default_factory=list)
    _last_ms: int = field(default=-10**15)

    def record(self, metrics: Optional[SpreadMetrics], state: str, force: bool = False) -> None:
        if metrics is None:
            return
        if not force and metrics.time_ms - self._last_ms < self.min_interval_ms:
            return
        self._last_ms = metrics.time_ms
        self.points.append(
            {
                "t": int(metrics.time_ms),
                "z": float(metrics.zscore),
                "s": float(metrics.spread_rub),
                "mu": float(metrics.mean_rub),
                "beta": float(metrics.beta),
                "p1": float(metrics.micro_leg1),
                "p2": float(metrics.micro_leg2),
                "bid1": float(metrics.best_bid_leg1),
                "ask1": float(metrics.best_ask_leg1),
                "state": state,
            }
        )

    def __bool__(self) -> bool:  # позволяет писать ``if trace:``
        return True

    def to_list(self) -> List[Dict[str, float]]:
        return list(self.points)


def _prime_bands(ax, start: datetime, end: datetime) -> None:
    """Подсветка активных окон входа (PrimeHoursOnly)."""
    day = start.date()
    first = True
    while day <= end.date():
        for begin, finish in PRIME_WINDOWS:
            left = max(datetime.combine(day, begin), start)
            right = min(datetime.combine(day, finish), end)
            if right > left:
                ax.axvspan(
                    left,
                    right,
                    color=COLORS["accent"],
                    alpha=0.07,
                    lw=0,
                    label="окно входа (Prime)" if first else None,
                )
                first = False
        day = day.fromordinal(day.toordinal() + 1)


def _markers(ax, trades: Sequence[TradeRecord], price_attr: str, is_exit: bool) -> None:
    """Точки входов (треугольники) и выходов (крестики) на ценовом графике."""
    for trade in trades:
        side = trade.side_leg1
        if is_exit:
            ax.scatter(
                _as_naive(trade.closed_ms),
                float(trade.leg1_exit_price),
                marker="X",
                s=90,
                color=EXIT_COLORS.get(trade.exit_reason.value, COLORS["neutral"]),
                edgecolors="white",
                linewidths=1.5,
                zorder=6,
            )
            continue
        marker = "^" if side is Side.BUY else "v"
        color = COLORS["buy"] if side is Side.BUY else COLORS["sell"]
        ax.scatter(
            _as_naive(trade.opened_ms),
            float(trade.leg1_entry_price),
            marker=marker,
            s=95,
            color=color,
            edgecolors="white",
            linewidths=0.9,
            zorder=6,
        )


def _exit_legend(trades: Sequence[TradeRecord]) -> List[Patch]:
    kinds = []
    for trade in trades:
        if trade.exit_reason.value not in kinds:
            kinds.append(trade.exit_reason.value)
    return [
        Patch(color=EXIT_COLORS.get(kind, COLORS["neutral"]), label=kind) for kind in kinds
    ]


def _kpi_lines(result: Any, preset_label: str) -> List[str]:
    totals = result.totals
    return [
        f"{preset_label} · {result.topology.value}",
        f"сделок {totals.trades} · дней {totals.profitable_days}/{totals.total_days} в плюсе",
        f"gross {totals.gross_pnl_rub:+.2f} RUB · fees {totals.fees_rub:.2f} · "
        f"net {totals.net_pnl_rub:+.2f} RUB",
        f"PF {totals.profit_factor:.2f} · Sharpe {totals.sharpe_annual:.2f} · "
        f"max DD {totals.max_drawdown_rub:.2f} RUB",
    ]


# --------------------------------------------------------------------- графики
def _split_gaps(times: Sequence[datetime], values: Sequence[float],
                gap_minutes: int = 20) -> Tuple[List[Optional[datetime]], List[Optional[float]]]:
    """Вставляет разрыв в линию на ночных/выходных паузах (чтобы не было прямых мостов)."""
    out_t: List[Optional[datetime]] = []
    out_v: List[Optional[float]] = []
    for index, (moment, value) in enumerate(zip(times, values)):
        if index and (moment - times[index - 1]).total_seconds() > gap_minutes * 60:
            out_t.append(None)
            out_v.append(None)
        out_t.append(moment)
        out_v.append(value)
    return out_t, out_v


def _chart_overview(result: Any, trace: Sequence[dict], trades: Sequence[TradeRecord], path: Path,
                    preset_label: str) -> Path:
    if not trace:
        raise RuntimeError("Нет траектории метрик: график обзора невозможен")
    times = [_as_naive(point["t"]) for point in trace]
    zs = [point["z"] for point in trace]
    p1 = [point["p1"] for point in trace]
    p2 = [point["p2"] for point in trace]
    params = result.params

    fig, axes = plt.subplots(3, 1, figsize=(15.5, 11.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0, 1.15]})
    ax1, ax2, ax3 = axes
    _prime_bands(ax1, times[0], times[-1])
    line_t, line_p1 = _split_gaps(times, p1)
    ax1.plot(line_t, line_p1, color=COLORS["leg1"], lw=1.0, label=f"Leg1 {result.preset_name.split('_')[0]}")
    _markers(ax1, trades, "leg1_entry_price", is_exit=False)
    _markers(ax1, trades, "leg1_exit_price", is_exit=True)
    ax1.set_ylabel("Leg1, RUB за акцию")
    ax1.grid(True, ls=":", lw=0.6)
    ax1.legend(loc="upper left", fontsize=8, ncols=2)
    ax1.set_title("Входы (▲ покупка / ▼ продажа) и выходы (✕) по Leg1", loc="left", fontsize=11)

    line_t2, line_p2 = _split_gaps(times, p2)
    ax2.plot(line_t2, line_p2, color=COLORS["leg2"], lw=1.0, label="Leg2 (ведущая нога)")
    ax2.ticklabel_format(useOffset=False, axis="y")
    ax2.set_ylabel("Leg2, RUB за акцию")
    ax2.grid(True, ls=":", lw=0.6)
    ax2.legend(loc="upper left", fontsize=8)

    _prime_bands(ax3, times[0], times[-1])
    zt, zv = _split_gaps(times, zs)
    ax3.plot(zt, zv, color=COLORS["zline"], lw=0.9, label="Z-score спреда")
    for level, style, label in (
        (params.z_entry, "--", f"Z_entry {params.z_entry:.2f}"),
        (-params.z_entry, "--", None),
        (params.z_exit, ":", f"Z_exit {params.z_exit:.2f}"),
        (-params.z_exit, ":", None),
        (params.z_stop, "-.", f"Z_stop {params.z_stop:.2f}"),
        (-params.z_stop, "-.", None),
    ):
        ax3.axhline(level, color=COLORS["neutral"], ls=style, lw=0.8, alpha=0.8, label=label)
    ax3.scatter(
        [_as_naive(trade.opened_ms) for trade in trades],
        [trade.entry_z for trade in trades],
        marker="^",
        s=55,
        color=COLORS["accent"],
        edgecolors="white",
        linewidths=0.5,
        zorder=6,
        label="вход (Z на входе)",
    )
    ax3.scatter(
        [_as_naive(trade.closed_ms) for trade in trades],
        [trade.exit_z for trade in trades],
        marker="X",
        s=45,
        color=COLORS["sell"],
        edgecolors="white",
        linewidths=0.5,
        zorder=6,
        label="выход (Z на выходе)",
    )
    ax3.set_ylabel("Z-score")
    ax3.set_ylim(-max(params.z_stop * 1.6, 4.0), max(params.z_stop * 1.6, 4.0))
    ax3.grid(True, ls=":", lw=0.6)
    ax3.legend(loc="upper left", fontsize=7, ncols=3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=14))

    for line in _kpi_lines(result, preset_label):
        pass
    fig.suptitle(" · ".join(_kpi_lines(result, preset_label)), x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=135)
    plt.close(fig)
    return path


def _chart_equity(result: Any, trades: Sequence[TradeRecord], path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 7.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1.4, 1.0]})
    if trades:
        times = [_as_naive(trade.closed_ms) for trade in trades]
        equity = []
        running = Decimal("0")
        for trade in trades:
            running += trade.net_pnl_rub
            equity.append(float(running))
        ax1.step(times, equity, where="post", color=COLORS["accent"], lw=1.7, label="эквити (net)")
        ax1.fill_between(times, equity, 0, step="post", color=COLORS["accent"], alpha=0.09)
        ax1.scatter(times, equity, s=16, color=COLORS["accent"], zorder=4)
        ax1.axhline(0, color=COLORS["neutral"], lw=0.8, ls=":")
        peak, drawdown, dd_series = equity[0], 0.0, []
        for value in equity:
            peak = max(peak, value)
            drawdown = min(drawdown, value - peak)
            dd_series.append(drawdown)
        ax2.fill_between(times, dd_series, 0, step="post", color=COLORS["loss"], alpha=0.12)
        ax2.step(times, dd_series, where="post", color=COLORS["loss"], lw=1.3, label="просадка")
        ax1.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="lower left", fontsize=8)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=12))
    ax1.set_ylabel("Накопленный net, RUB")
    ax2.set_ylabel("Просадка, RUB")
    for ax in (ax1, ax2):
        ax.grid(True, ls=":", lw=0.6)
    ax1.set_title("Кривая эквити одной позицией (1 лот)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=135)
    plt.close(fig)
    return path


def _chart_daily(result: Any, path: Path) -> Path:
    summaries = list(result.summaries)
    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    labels = [summary.date for summary in summaries]
    values = [float(summary.net_pnl_rub) for summary in summaries]
    positions = range(len(labels))
    colors = [COLORS["profit"] if value >= 0 else COLORS["loss"] for value in values]
    ax.bar(positions, values, color=colors, alpha=0.85, width=0.72)
    cumulative = []
    running = 0.0
    for value in values:
        running += value
        cumulative.append(running)
    ax2 = ax.twinx()
    ax2.plot(positions, cumulative, color=COLORS["accent"], lw=1.5, marker="o", ms=2.5)
    ax2.set_ylabel("Накопленный net, RUB", color=COLORS["accent"])
    ax2.tick_params(axis="y", colors=COLORS["accent"])
    ax.set_xticks(list(positions))
    step = max(len(labels) // 12, 1)
    ax.set_xticks(list(positions)[::step])
    ax.set_xticklabels([label[5:] for label in labels][::step], rotation=45, fontsize=7, ha="right")
    ax.set_ylabel("Net за день, RUB")
    ax.grid(True, axis="y", ls=":", lw=0.6)
    green = sum(1 for value in values if value > 0)
    ax.set_title(
        f"Дневной PnL: зелёных {green} из {len(values)} "
        f"({green / max(len(values), 1) * 100:.0f}%)",
        loc="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=135)
    plt.close(fig)
    return path


def _chart_trades(result: Any, trades: Sequence[TradeRecord], path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8), gridspec_kw={"width_ratios": [1.7, 1.0]})
    if trades:
        nets = [float(trade.net_pnl_rub) for trade in trades]
        colors = [EXIT_COLORS.get(trade.exit_reason.value, COLORS["neutral"]) for trade in trades]
        ax1.bar(range(len(nets)), nets, color=colors, width=0.85)
        ax1.axhline(0, color=COLORS["neutral"], lw=0.8)
        ax1.set_title("Результат каждой сделки (цвет = тип выхода)", loc="left", fontsize=10)
        ax1.set_xlabel("номер сделки")
        ax1.set_ylabel("net, RUB")
        ax1.legend(handles=_exit_legend(trades), fontsize=7, loc="lower right")
        ax1.grid(True, axis="y", ls=":", lw=0.6)

        for kind, color in EXIT_COLORS.items():
            subset = [trade for trade in trades if trade.exit_reason.value == kind]
            if not subset:
                continue
            ax2.scatter(
                [float(trade.mfe_rub) for trade in subset],
                [float(trade.net_pnl_rub) for trade in subset],
                color=color,
                s=38,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.4,
                label=kind,
            )
        ax2.axhline(0, color=COLORS["neutral"], lw=0.8, ls=":")
        ax2.set_xlabel("MFE сделки, RUB")
        ax2.set_ylabel("фактический net, RUB")
        ax2.set_title("Сколько от максимума удалось забрать", loc="left", fontsize=10)
        ax2.legend(fontsize=7)
        ax2.grid(True, ls=":", lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=135)
    plt.close(fig)
    return path


def _chart_zoom(result: Any, trade: TradeRecord, candles1: Sequence[Candle],
                candles2: Sequence[Candle], path: Path, pad_minutes: int = 25) -> Path:
    lo = trade.opened_ms - pad_minutes * 60_000
    hi = trade.closed_ms + pad_minutes * 60_000
    window1 = [candle for candle in candles1 if lo <= candle.time_ms <= hi]
    window2 = [candle for candle in candles2 if lo <= candle.time_ms <= hi]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.0, 6.6), sharex=True)

    def draw_candles(ax, candles, color, label):
        for candle in candles:
            moment = _as_naive(candle.time_ms)
            up = candle.close >= candle.open
            ax.vlines(moment, float(candle.low), float(candle.high),
                      color=color if up else COLORS["neutral"], lw=0.7)
            ax.vlines(moment, float(candle.open), float(candle.close),
                      color=color if up else COLORS["neutral"], lw=2.6, alpha=0.9)
        ax.plot([_as_naive(candle.time_ms) for candle in candles],
                [float(candle.close) for candle in candles],
                color=color, lw=0.7, alpha=0.5, label=label)

    draw_candles(ax1, window1, COLORS["leg1"], "Leg1 (1 мин)")
    entry_color = COLORS["buy"] if trade.side_leg1 is Side.BUY else COLORS["sell"]
    ax1.axvline(_as_naive(trade.opened_ms), color=entry_color, lw=1.4, ls="--")
    ax1.axvline(_as_naive(trade.closed_ms), color=EXIT_COLORS.get(trade.exit_reason.value, COLORS["neutral"]),
                lw=1.4, ls="--")
    ax1.axhline(float(trade.leg1_entry_price), color=entry_color, lw=0.9, ls=":",
                label=f"вход ({trade.side_leg1.value}) {trade.leg1_entry_price}")
    ax1.axhline(float(trade.leg1_exit_price), color=COLORS["neutral"], lw=0.9, ls="--",
                label=f"выход {trade.leg1_exit_price}")
    ax1.scatter([_as_naive(trade.opened_ms)], [float(trade.leg1_entry_price)],
                marker="^" if trade.side_leg1 is Side.BUY else "v", s=150,
                color=entry_color, edgecolors="white", linewidths=1.2, zorder=6)
    ax1.scatter([_as_naive(trade.closed_ms)], [float(trade.leg1_exit_price)], marker="X", s=110,
                color=EXIT_COLORS.get(trade.exit_reason.value, COLORS["neutral"]),
                edgecolors="white", zorder=6)
    ax1.set_ylabel("Leg1, RUB")
    ax1.ticklabel_format(useOffset=False, axis="y")
    ax1.grid(True, ls=":", lw=0.6)
    ax1.legend(fontsize=7, loc="best")

    draw_candles(ax2, window2, COLORS["leg2"], "Leg2 (1 мин)")
    ax2.axvline(_as_naive(trade.opened_ms), color=COLORS["buy"], lw=1.2, ls="--")
    ax2.axvline(_as_naive(trade.closed_ms), color=COLORS["sell"], lw=1.2, ls="--")
    ax2.set_ylabel("Leg2, RUB")
    ax2.ticklabel_format(useOffset=False, axis="y")
    ax2.grid(True, ls=":", lw=0.6)
    ax2.legend(fontsize=7, loc="best")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))

    ax1.set_title(
        f"Сделка #{trade.trade_id}: {trade.side_leg1.value} 1 лот · вход "
        f"{ms_to_msk(trade.opened_ms):%d.%m %H:%M:%S} · выход "
        f"{ms_to_msk(trade.closed_ms):%d.%m %H:%M:%S} ({trade.exit_reason.value}) · "
        f"net {trade.net_pnl_rub:+.3f} RUB",
        loc="left",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=135)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- отчёты
def _trades_rows(trades: Sequence[TradeRecord]) -> List[List[str]]:
    rows = []
    for trade in trades:
        rows.append(
            [
                str(trade.trade_id),
                ms_to_msk(trade.opened_ms).strftime("%d.%m %H:%M:%S"),
                trade.side_leg1.value,
                str(trade.leg1_entry_price),
                ms_to_msk(trade.closed_ms).strftime("%d.%m %H:%M:%S"),
                str(trade.leg1_exit_price),
                trade.exit_reason.value,
                f"{trade.hold_seconds / 60:.0f}",
                f"{trade.entry_z:+.2f}",
                f"{trade.exit_z:+.2f}",
                f"{trade.mfe_rub:+.3f}",
                f"{trade.mae_rub:+.3f}",
                f"{trade.gross_pnl_rub:+.3f}",
                f"{trade.fees_rub:.3f}",
                f"{trade.net_pnl_rub:+.3f}",
            ]
        )
    return rows


def write_trades_csv(path: Path, trades: Sequence[TradeRecord]) -> Path:
    header = [
        "id", "открытие", "сторона", "цена входа", "закрытие", "цена выхода", "причина выхода",
        "мин в позиции", "Z вход", "Z выход", "MFE", "MAE", "gross", "fees", "net",
    ]
    lines = [";".join(header)]
    lines.extend(";".join(row) for row in _trades_rows(trades))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_index_html(result: Any, charts: Sequence[Path], out_dir: Path, preset_label: str,
                     trades: Sequence[TradeRecord], trace_points: int) -> Path:
    totals = result.totals
    rows = "\n".join(
        "<tr>"
        + "".join(f"<td>{cell}</td>" for cell in row)
        + "</tr>"
        for row in _trades_rows(trades)
    )
    images = "\n".join(
        f'<figure><img src="{chart.name}" alt="{chart.stem}"><figcaption>{chart.name}</figcaption></figure>'
        for chart in charts
    )
    net = float(totals.net_pnl_rub)
    color = "#2ecc71" if net >= 0 else "#ff5c5c"
    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Bot A — визуализация работы · {preset_label}</title>
<style>
 body{{background:#0f1419;color:#d7e0ea;font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:24px}}
 h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:28px 0 8px;color:#9fb3c8}}
 .kpis{{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 6px}}
 .kpi{{background:#161d26;border:1px solid #243040;border-radius:8px;padding:10px 14px;min-width:150px}}
 .kpi b{{display:block;font-size:17px;margin-top:3px}}
 .muted{{color:#8b9aab;font-size:12px}}
 figure{{margin:0 0 22px}} img{{width:100%;border-radius:8px;border:1px solid #243040}}
 figcaption{{color:#8b9aab;font-size:11px;margin-top:5px}}
 table{{border-collapse:collapse;font-size:11.5px;width:100%}}
 th,td{{border-bottom:1px solid #243040;padding:4px 7px;text-align:right;white-space:nowrap}}
 th{{color:#9fb3c8;text-align:right;position:sticky;top:0;background:#161d26}}
 td:nth-child(2),td:nth-child(3),td:nth-child(5),td:nth-child(7){{text-align:left}}
 .wrap{{max-height:520px;overflow:auto;border:1px solid #243040;border-radius:8px}}
</style></head><body>
<h1>Bot A — как работает бот на реальных данных MOEX</h1>
<div class="muted">пресет {preset_label} · топология {result.topology.value} ·
 Z_Entry {result.params.z_entry} · Z_Exit {result.params.z_exit} · Z_Stop {result.params.z_stop} ·
 EWMA {result.params.ewma_window} · MaxHold {result.params.max_hold_seconds} c ·
 точек траектории {trace_points}</div>
<div class="kpis">
 <div class="kpi">Сделок<b>{totals.trades}</b></div>
 <div class="kpi">Net PnL<b style="color:{color}">{net:+.2f} RUB</b></div>
 <div class="kpi">Gross / комиссии<b>{float(totals.gross_pnl_rub):+.2f} / {float(totals.fees_rub):.2f}</b></div>
 <div class="kpi">Дней в плюсе<b>{totals.profitable_days}/{totals.total_days}</b></div>
 <div class="kpi">Profit Factor<b>{totals.profit_factor:.2f}</b></div>
 <div class="kpi">Max DD<b>{float(totals.max_drawdown_rub):.2f} RUB</b></div>
 <div class="kpi">Sharpe (год.)<b>{totals.sharpe_annual:.2f}</b></div>
 <div class="kpi">Ср. удержание<b>{totals.avg_hold_seconds / 60:.0f} мин</b></div>
</div>
<h2>Графики</h2>
{images}
<h2>Все сделки: когда вошёл, когда вышел</h2>
<div class="wrap"><table>
<tr><th>#</th><th>открытие</th><th>сторона</th><th>цена входа</th><th>закрытие</th>
<th>цена выхода</th><th>причина</th><th>мин</th><th>Z вх.</th><th>Z вых.</th>
<th>MFE</th><th>MAE</th><th>gross</th><th>fees</th><th>net</th></tr>
{rows}
</table></div>
</body></html>
"""
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def render_charts(
    result: Any,
    trace: Optional[TraceRecorder],
    out_dir: str | Path,
    stem: str,
    preset_label: Optional[str] = None,
    candles1: Optional[Sequence[Candle]] = None,
    candles2: Optional[Sequence[Candle]] = None,
    zoom_trades: int = 3,
    logger: Optional[logging.Logger] = None,
) -> List[Path]:
    """Рисует полный набор графиков и HTML-страницу. Возвращает список файлов."""
    log = logger or logging.getLogger(LOGGER_NAME)
    _style()
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    label = preset_label or result.preset_name
    trades = list(result.trades)
    charts: List[Path] = []

    if trace is not None and trace.points:
        charts.append(_chart_overview(result, trace.points, trades, directory / "01_overview.png", label))
    charts.append(_chart_equity(result, trades, directory / "02_equity.png"))
    charts.append(_chart_daily(result, directory / "03_daily_pnl.png"))
    charts.append(_chart_trades(result, trades, directory / "04_trades_and_exits.png"))

    if candles1 and candles2 and trades:
        ordered = sorted(trades, key=lambda trade: abs(float(trade.net_pnl_rub)), reverse=True)[:zoom_trades]
        for index, trade in enumerate(sorted(ordered, key=lambda trade: trade.opened_ms), start=1):
            charts.append(
                _chart_zoom(
                    result,
                    trade,
                    candles1,
                    candles2,
                    directory / f"05_zoom_{index:02d}_trade_{trade.trade_id}.png",
                )
            )

    write_trades_csv(directory / "trades.csv", trades)
    index = write_index_html(
        result, charts, directory, label, trades, len(trace.points) if trace is not None else 0
    )
    charts.append(index)

    meta = {
        "preset": result.preset_name,
        "topology": result.topology.value,
        "trades": result.totals.trades,
        "netPnlRub": str(result.totals.net_pnl_rub),
        "grossPnlRub": str(result.totals.gross_pnl_rub),
        "feesRub": str(result.totals.fees_rub),
        "charts": [chart.name for chart in charts],
    }
    (directory / "charts.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Графики сохранены: %s (%d файлов)", directory, len(charts))
    return charts
