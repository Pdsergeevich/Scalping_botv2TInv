"""Отчёты бэктеста: консольные таблицы, Markdown, CSV и JSON."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from bot_a.models.domain import BacktestTotals, DailyBacktestSummary, TradeRecord
from bot_a.utils.quotation import fmt_rub


def _fmt_money(value: Any) -> str:
    return fmt_rub(value)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} c"
    minutes = seconds / 60.0
    return f"{minutes:.1f} мин"


def render_daily_table(summaries: Sequence[DailyBacktestSummary]) -> str:
    """Таблица «день за днём» с PnL, комиссией и просадкой."""
    header = (
        f"{'Дата':<12}{'Сделок':>7}{'Win %':>8}{'Gross RUB':>14}{'Fees RUB':>12}"
        f"{'Net RUB':>14}{'Net % ГО':>10}{'Max DD RUB':>13}{'Ср. удерж.':>12}"
    )
    lines = [header, "-" * len(header)]
    for summary in summaries:
        lines.append(
            f"{summary.date:<12}"
            f"{summary.trades:>7d}"
            f"{summary.win_rate_pct:>8.1f}"
            f"{_fmt_money(summary.gross_pnl_rub):>14}"
            f"{_fmt_money(summary.fees_rub):>12}"
            f"{_fmt_money(summary.net_pnl_rub):>14}"
            f"{summary.net_pnl_pct_of_margin:>9.2f}%"
            f"{_fmt_money(summary.max_drawdown_rub):>13}"
            f"{_fmt_duration(summary.avg_hold_seconds):>12}"
        )
    if not summaries:
        lines.append("(нет сделок за выбранный период)")
    return "\n".join(lines)


def render_totals(totals: BacktestTotals) -> str:
    ratio = f"{totals.profitable_days}/{totals.total_days}" if totals.total_days else "0/0"
    lines = [
        f"Всего сделок:            {totals.trades}",
        f"Прибыльных дней:         {ratio}",
        f"Валовой PnL:             {_fmt_money(totals.gross_pnl_rub)} RUB",
        f"Комиссии T-Invest:       {_fmt_money(totals.fees_rub)} RUB",
        f"Чистый PnL:              {_fmt_money(totals.net_pnl_rub)} RUB",
        f"Win Rate:                {totals.win_rate_pct:.1f}%",
        f"Profit Factor:           {totals.profit_factor:.2f}",
        f"Sharpe (годовой):        {totals.sharpe_annual:.2f}",
        f"Среднее удержание:       {_fmt_duration(totals.avg_hold_seconds)}",
        f"Fee drag:                {totals.fee_drag_pct:.1f}% от валового PnL",
        f"Максимальная просадка:   {_fmt_money(totals.max_drawdown_rub)} RUB",
    ]
    return "\n".join(lines)


def render_console_report(
    title: str,
    summaries: Sequence[DailyBacktestSummary],
    totals: BacktestTotals,
    context: Mapping[str, Any] | None = None,
) -> str:
    blocks: List[str] = [title, "=" * len(title)]
    if context:
        for key, value in context.items():
            blocks.append(f"{key}: {value}")
        blocks.append("")
    blocks.append(render_daily_table(summaries))
    blocks.append("")
    blocks.append("Сводные метрики")
    blocks.append("---------------")
    blocks.append(render_totals(totals))
    return "\n".join(blocks)


def render_markdown(
    title: str,
    summaries: Sequence[DailyBacktestSummary],
    totals: BacktestTotals,
    context: Mapping[str, Any] | None = None,
) -> str:
    lines: List[str] = [f"# {title}", ""]
    if context:
        lines.append("## Параметры прогона")
        lines.append("")
        lines.append("| Параметр | Значение |")
        lines.append("| --- | --- |")
        for key, value in context.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
    lines.append("## День за днём")
    lines.append("")
    lines.append("| Дата | Сделок | Win Rate, % | Gross, RUB | Fees, RUB | Net, RUB | Net, % ГО | Max DD, RUB | Ср. удержание |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in summaries:
        lines.append(
            "| {date} | {trades} | {win:.1f} | {gross} | {fees} | {net} | {net_pct:.2f} | {dd} | {hold} |".format(
                date=summary.date,
                trades=summary.trades,
                win=summary.win_rate_pct,
                gross=_fmt_money(summary.gross_pnl_rub),
                fees=_fmt_money(summary.fees_rub),
                net=_fmt_money(summary.net_pnl_rub),
                net_pct=summary.net_pnl_pct_of_margin,
                dd=_fmt_money(summary.max_drawdown_rub),
                hold=_fmt_duration(summary.avg_hold_seconds),
            )
        )
    lines.append("")
    lines.append("## Сводные метрики")
    lines.append("")
    lines.append("| Метрика | Значение |")
    lines.append("| --- | ---: |")
    lines.append(f"| Всего сделок | {totals.trades} |")
    lines.append(f"| Прибыльных дней | {totals.profitable_days}/{totals.total_days} |")
    lines.append(f"| Валовой PnL, RUB | {_fmt_money(totals.gross_pnl_rub)} |")
    lines.append(f"| Комиссии, RUB | {_fmt_money(totals.fees_rub)} |")
    lines.append(f"| Чистый PnL, RUB | {_fmt_money(totals.net_pnl_rub)} |")
    lines.append(f"| Win Rate, % | {totals.win_rate_pct:.1f} |")
    lines.append(f"| Profit Factor | {totals.profit_factor:.2f} |")
    lines.append(f"| Sharpe (годовой) | {totals.sharpe_annual:.2f} |")
    lines.append(f"| Среднее удержание | {_fmt_duration(totals.avg_hold_seconds)} |")
    lines.append(f"| Fee drag, % | {totals.fee_drag_pct:.1f} |")
    lines.append(f"| Максимальная просадка, RUB | {_fmt_money(totals.max_drawdown_rub)} |")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: str | Path, summaries: Iterable[DailyBacktestSummary]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "date",
                "trades",
                "wins",
                "win_rate_pct",
                "gross_pnl_rub",
                "fees_rub",
                "net_pnl_rub",
                "net_pnl_pct_of_margin",
                "max_drawdown_rub",
                "avg_hold_seconds",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.date,
                    summary.trades,
                    summary.wins,
                    f"{summary.win_rate_pct:.2f}",
                    f"{summary.gross_pnl_rub:.2f}",
                    f"{summary.fees_rub:.2f}",
                    f"{summary.net_pnl_rub:.2f}",
                    f"{summary.net_pnl_pct_of_margin:.4f}",
                    f"{summary.max_drawdown_rub:.2f}",
                    f"{summary.avg_hold_seconds:.1f}",
                ]
            )
    return target


def write_trades_csv(path: str | Path, trades: Sequence[TradeRecord]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "trade_id",
                "opened_ms",
                "closed_ms",
                "topology",
                "side_leg1",
                "quantity",
                "leg1_entry",
                "leg1_exit",
                "leg2_entry",
                "leg2_exit",
                "gross_pnl_rub",
                "fees_rub",
                "net_pnl_rub",
                "mfe_rub",
                "mae_rub",
                "entry_z",
                "exit_z",
                "margin_rub",
                "exit_reason",
            ]
        )
        for trade in trades:
            writer.writerow(
                [
                    trade.trade_id,
                    trade.opened_ms,
                    trade.closed_ms,
                    trade.topology.value,
                    trade.side_leg1.value,
                    trade.quantity,
                    f"{trade.leg1_entry_price}",
                    f"{trade.leg1_exit_price}",
                    f"{trade.leg2_entry_price}" if trade.leg2_entry_price is not None else "",
                    f"{trade.leg2_exit_price}" if trade.leg2_exit_price is not None else "",
                    f"{trade.gross_pnl_rub:.2f}",
                    f"{trade.fees_rub:.2f}",
                    f"{trade.net_pnl_rub:.2f}",
                    f"{trade.mfe_rub:.2f}",
                    f"{trade.mae_rub:.2f}",
                    f"{trade.entry_z:.3f}",
                    f"{trade.exit_z:.3f}",
                    f"{trade.margin_rub:.2f}",
                    trade.exit_reason.value,
                ]
            )
    return target


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return target


def write_reports(
    reports_dir: str | Path,
    stem: str,
    title: str,
    summaries: Sequence[DailyBacktestSummary],
    totals: BacktestTotals,
    context: Mapping[str, Any] | None = None,
    trades: Sequence[TradeRecord] = (),
    extra: Mapping[str, Any] | None = None,
) -> List[str]:
    """Сохраняет Markdown, CSV по дням, CSV по сделкам и JSON-сводку."""
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    markdown_path = directory / f"{stem}.md"
    markdown_path.write_text(render_markdown(title, summaries, totals, context), encoding="utf-8")
    paths.append(str(markdown_path))

    paths.append(str(write_csv(directory / f"{stem}_daily.csv", summaries)))
    if trades:
        paths.append(str(write_trades_csv(directory / f"{stem}_trades.csv", trades)))

    payload: Dict[str, Any] = {
        "title": title,
        "context": dict(context or {}),
        "daily": [asdict(summary) for summary in summaries],
        "totals": asdict(totals),
    }
    if extra:
        payload.update(extra)
    paths.append(str(write_json(directory / f"{stem}.json", payload)))
    return paths
