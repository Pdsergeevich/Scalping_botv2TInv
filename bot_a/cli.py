"""CLI Bot A: preflight | backtest | optimize | record | paper | live."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from datetime import datetime, time
from typing import Optional, Sequence

from bot_a.backtest.candles import TInvestCandleSource, resample_candles
from bot_a.backtest.grid_search import GridSearchOptimizer
from bot_a.backtest.historical_candles import HistoricalCandleBacktester
from bot_a.backtest.iss_source import IssCandleSource, fetch_futures_metadata, fetch_stock_metadata
from bot_a.backtest.recorder import RecordingService
from bot_a.backtest.tick_replay import ReplayRange, TickReplayBacktester, build_replay_source
from bot_a.broker import tinvest as broker
from bot_a.config import AppConfig
from bot_a.economics.fees import Tariff
from bot_a.engine.fsm import StrategyParams
from bot_a.logging_setup import configure_logging
from bot_a.models.domain import ExecutionTopology
from bot_a.services.live_worker import LiveTradingWorker
from bot_a.services.preflight import run_preflight
from bot_a.utils.time_provider import MSK, msk_to_ms

MODES = ("preflight", "backtest", "optimize", "record", "paper", "live", "watch")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bot_a",
        description="Bot A — парный скальпинг/стат-арбитраж на MOEX через T-Invest API v2",
    )
    parser.add_argument("--mode", required=True, choices=MODES, help="режим работы")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["pairs", "breakout"],
        help="торговое ядро: pairs (парный возврат к среднему, v2.x) или breakout (пробой сжатия, v3.0)",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="инструмент для --strategy breakout (например CNYRUBF); по умолчанию activeBreakout",
    )
    parser.add_argument("--config", default="appsettings.json", help="путь к appsettings.json")
    parser.add_argument("--preset", default=None, help="имя пресета пары из конфигурации")
    parser.add_argument("--token", default=None, help="токен T-Invest (переопределяет конфигурацию)")
    parser.add_argument("--account", default=None, help="идентификатор счёта")
    parser.add_argument("--tariff", default=None, choices=[item.value for item in Tariff], help="тариф брокера")
    parser.add_argument(
        "--topology",
        default=None,
        choices=[item.value for item in ExecutionTopology],
        help="топология исполнения",
    )
    parser.add_argument("--z-entry", type=float, default=None, help="порог входа по |Z|")
    parser.add_argument("--z-exit", type=float, default=None, help="порог выхода по |Z|")
    parser.add_argument("--ewma-window", type=int, default=None, help="окно EWMA (тиков)")
    parser.add_argument(
        "--macro-window",
        type=int,
        default=None,
        help="макро-окно беты для дрейфа (замеров; для --synth-mode close масштабируется вниз)",
    )
    parser.add_argument("--quantity", type=int, default=None, help="количество лотов на вход")
    parser.add_argument("--days", type=int, default=7, help="глубина истории для backtest/optimize")
    parser.add_argument(
        "--bar-minutes",
        dest="bar_minutes",
        type=int,
        default=1,
        help="таймфрейм пробойной стратегии в минутах (1 по умолчанию; v3.1 рекомендует 30)",
    )
    parser.add_argument(
        "--entry-mode",
        dest="entry_mode",
        default=None,
        choices=["bar_close", "stop_level"],
        help="вход по закрытию бара пробоя или по касанию уровня H+1 шаг (v3.1)",
    )
    parser.add_argument(
        "--overnight",
        dest="overnight",
        action="store_true",
        default=None,
        help="разрешить перенос позиции через вечернюю сессию и ночь (v3.1)",
    )
    parser.add_argument(
        "--margin-utilization",
        dest="margin_utilization",
        type=float,
        default=60.0,
        help="максимальная доля капитала под ГО для --ticker ALL_SWING (по умолчанию 60 %%)",
    )
    parser.add_argument(
        "--no-overnight",
        dest="overnight",
        action="store_false",
        help="только интрадей: закрывать позиции в 18:35 MSK",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=50000.0,
        help="капитал для сайзинга пробойной стратегии (RUB, по умолчанию 50000)",
    )
    parser.add_argument(
        "--risk-pct",
        dest="risk_pct",
        type=float,
        default=1.5,
        help="риск на сделку в %% капитала для --strategy breakout (по умолчанию 1.5)",
    )
    parser.add_argument("--from", dest="date_from", default=None, help="начало периода (YYYY-MM-DD, MSK)")
    parser.add_argument("--to", dest="date_to", default=None, help="конец периода (YYYY-MM-DD, MSK)")
    parser.add_argument("--interval", default="1min", choices=["1min", "5min", "15min", "1hour"], help="интервал свечей")
    parser.add_argument("--source", default="api",
        choices=["api", "iss", "recordings"],
        help="источник данных бэктеста: T-Invest API, MOEX ISS (без токена) или записи",
    )
    parser.add_argument(
        "--synth-mode",
        default="ticks",
        choices=["ticks", "close"],
        help="режим синтеза: ticks (12 подтиков на минуту) или close (только закрытия 1-мин баров)",
    )
    parser.add_argument("--recordings", default=None, help="путь к .jsonl.gz или каталогу записей")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="число процессов для optimize (по умолчанию 2, максимум 4 — чтобы не грузить CPU)",
    )
    parser.add_argument(
        "--record-interval-ms",
        type=int,
        default=150,
        help="интервал поллинга L2 в режиме record (REST; ISS-фолбэк не чаще 1000 мс)",
    )
    parser.add_argument("--record-depth", type=int, default=20, help="глубина стакана для record")
    parser.add_argument(
        "--port",
        type=int,
        default=8099,
        help="порт live-дашборда для --mode watch (по умолчанию 8099)",
    )
    parser.add_argument(
        "--record-l2",
        dest="record_l2",
        action="store_true",
        default=None,
        help="писать живой L2 в recordings/ (по умолчанию включено для --preset ALL_ACTIVE)",
    )
    parser.add_argument(
        "--no-record-l2",
        dest="record_l2",
        action="store_false",
        help="не писать L2 на диск",
    )
    parser.add_argument("--duration", type=float, default=None, help="длительность работы, секунды")
    parser.add_argument("--log-level", default=None, help="уровень логирования (Information/Debug)")
    parser.add_argument("--report-stem", default=None, help="префикс файлов отчёта")
    parser.add_argument(
        "--charts",
        action="store_true",
        help="сохранить графики входов/выходов, Z-score, эквити и разбор сделок (PNG + HTML)",
    )
    parser.add_argument(
        "--chart-dir",
        default=None,
        help="каталог для графиков (по умолчанию reports/charts/<префикс>)",
    )
    parser.add_argument(
        "--chart-trades",
        type=int,
        default=3,
        help="сколько сделок показать в детальном приближении (по умолчанию 3)",
    )
    parser.add_argument("--no-report", action="store_true", help="не сохранять отчёты на диск")
    parser.add_argument("--json", action="store_true", help="вывести итог в JSON")
    parser.add_argument("--yes", action="store_true", help="подтверждение реальной торговли (--mode live)")
    parser.add_argument("--no-flatten", action="store_true", help="не закрывать позицию при остановке")
    parser.add_argument(
        "--monday-ready",
        action="store_true",
        help="дополнить preflight офлайн-проверкой готовности к сессии (monday-ready)",
    )
    return parser


def _apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    updated = config.with_overrides(
        tariff=Tariff(args.tariff) if args.tariff else None,
        active_preset=args.preset,
        token=args.token,
        account_id=args.account,
    )
    # Переопределения пресета (zEntry / topology / activeSessionWindows) идут до флагов CLI.
    # Для ALL_ACTIVE пресета нет — каждая пара берёт свои настройки внутри раннера.
    if (args.preset or "").upper() != "ALL_ACTIVE":
        updated = updated.with_preset_overrides()
    strategy: StrategyParams = updated.strategy
    if args.z_entry is not None:
        strategy = strategy.with_overrides(z_entry=float(args.z_entry))
    if args.z_exit is not None:
        strategy = strategy.with_overrides(z_exit=float(args.z_exit))
    if args.ewma_window is not None:
        strategy = strategy.with_overrides(ewma_window=int(args.ewma_window))
    if getattr(args, "macro_window", None) is not None:
        macro = int(args.macro_window)
        strategy = strategy.with_overrides(
            macro_beta_window=macro,
            macro_beta_min_samples=max(min(strategy.macro_beta_min_samples, macro // 2), 10),
        )
    if args.quantity is not None:
        strategy = strategy.with_overrides(quantity_lots=max(int(args.quantity), 1))
    updated = updated.with_overrides(
        topology=ExecutionTopology(args.topology) if args.topology else None,
        strategy=strategy,
    )
    if args.no_flatten:
        from dataclasses import replace

        updated = replace(updated, risk=replace(updated.risk, flatten_on_shutdown=False))
    if args.log_level:
        from dataclasses import replace

        updated = replace(updated, logging=replace(updated.logging, level=args.log_level))
    return updated


def _result_from_snapshot(config: AppConfig, preset, snapshot: dict, trades) -> "BacktestResult":
    """Собирает BacktestResult из живого снапшота ядра (для отчётов и графиков)."""
    from bot_a.backtest.historical_candles import BacktestResult
    from bot_a.engine.ledger import PortfolioLedger

    ledger = PortfolioLedger(config.execution.paper_initial_cash_rub)
    for trade in trades:
        ledger.register(trade)
    started = int(trades[0].opened_ms) if trades else 0
    finished = int(trades[-1].closed_ms) if trades else 0
    return BacktestResult(
        title=f"Paper на живом стакане {preset.name}",
        preset_name=preset.name,
        topology=config.execution.topology,
        params=config.strategy,
        summaries=tuple(ledger.daily_summaries()),
        totals=ledger.totals(),
        trades=tuple(trades),
        started_ms=started,
        finished_ms=finished,
        candles_leg1=0,
        candles_leg2=0,
        open_position_at_end=bool(snapshot.get("positionOpen")),
        report_paths=(),
        events_processed=int(snapshot.get("books") or 0),
        counters=dict(snapshot.get("counters") or {}),
    )


def _parse_period(args: argparse.Namespace) -> tuple[Optional[int], Optional[int]]:
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    if args.date_from:
        day = datetime.fromisoformat(args.date_from).date()
        start_ms = msk_to_ms(datetime.combine(day, time.min, tzinfo=MSK))
    if args.date_to:
        day = datetime.fromisoformat(args.date_to).date()
        end_ms = msk_to_ms(datetime.combine(day, time.max, tzinfo=MSK))
    return start_ms, end_ms


def _install_stop_handler(stop: asyncio.Event) -> None:
    """Ctrl+C -> мягкая остановка с отменой заявок, а не мгновенный kill."""
    loop = asyncio.get_running_loop()

    def handler(signum, frame) -> None:  # noqa: ARG001
        try:
            loop.call_soon_threadsafe(stop.set)
        except Exception:
            pass

    try:
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):
        pass


async def _resolve_iss_preset(preset):
    """Дополняет пресет реальными метаданными MOEX ISS (шаг, лот, ГО)."""
    def load(meta):
        loader = fetch_futures_metadata if meta.is_futures else fetch_stock_metadata
        return loader(meta.ticker)

    leg1, leg2 = await asyncio.gather(
        asyncio.to_thread(load, preset.leg1),
        asyncio.to_thread(load, preset.leg2),
    )
    return preset.with_metadata(leg1, leg2)


async def _run_all_active(args: argparse.Namespace, config: AppConfig, logger) -> int:
    """`--preset ALL_ACTIVE`: все включённые пары в одном процессе (v2.5)."""
    from bot_a.broker.tinkoff_rest_client import TInvestRestClient, TInvestRestError
    from bot_a.services.multi_preset import MultiPresetRunner

    names = config.trading_preset_names()
    if not names:
        logger.error("Нет включённых пар в appsettings.json — ALL_ACTIVE запускать нечего")
        return 2
    presets = [config.preset(name) for name in names]
    token = config.t_invest.resolve_token(config.base_dir)
    if not token:
        logger.error("Нужен токен T-Invest: --token, $env:TINVEST_TOKEN или secrets/tinvest_token.txt")
        return 3
    logger.info(
        "ALL_ACTIVE: пар %d (%s), режим %s, тариф %s",
        len(presets),
        ", ".join(names),
        args.mode,
        config.fees.tariff.value,
    )
    client = TInvestRestClient(
        token, config.t_invest.app_name, timeout=config.t_invest.request_timeout_seconds
    )
    stop = asyncio.Event()
    _install_stop_handler(stop)
    # v2.6.6: для ALL_ACTIVE живой L2 пишется на диск по умолчанию — иначе после
    # сессии нечем проверить, где исполнение между `ticks` и `close`.
    record_l2 = True if args.record_l2 is None else bool(args.record_l2)
    runner = MultiPresetRunner(
        config,
        presets,
        client,
        interval_ms=int(args.record_interval_ms),
        depth=int(args.record_depth),
        port=int(args.port),
        serve_dashboard=True,
        logger=logger,
        record_l2=record_l2,
    )
    print(f"Мульти-дашборд: http://127.0.0.1:{int(args.port)}/  (Ctrl+C — остановка)")
    try:
        result = await runner.run(duration_seconds=args.duration, stop_event=stop)
    except TInvestRestError as exc:
        logger.error("T-Invest REST недоступен: %s", exc)
        return 3
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for pair in result.get("pairs", []):
            print(
                f"{pair['name']}: сделок {pair['trades']}, net {pair['netRub']} RUB, "
                f"skewSkips {pair['skewSkips']}, ошибок {pair['restErrors']}"
            )
    return 0


async def _dispatch_swing_basket(args: argparse.Namespace, config: AppConfig, logger) -> int:
    """`--ticker ALL_SWING`: корзина свинга с общим счётом и общей маржой (v3.2)."""
    from bot_a.backtest.breakout_backtest import aggregate, portfolio_curve
    from bot_a.backtest.swing_portfolio import (
        SwingAccountBacktester,
        SwingAccountSettings,
    )
    from bot_a.engine.breakout_fsm import Bar
    from bot_a.services.iss_quotes import IssQuoteSource
    from bot_a.services.swing_runner import (
        SwingBasketRunner,
        SwingBasketSettings,
        basket_from_config,
    )
    from bot_a.utils.time_provider import MSK

    tickers = basket_from_config(config)
    if not tickers:
        logger.error("Корзина свинга пуста: заполните swingBasket или breakoutPresets")
        return 2
    bar_minutes = max(int(getattr(args, "bar_minutes", 1) or 1), 1)
    capital = float(getattr(args, "capital", 50_000.0))
    risk = float(getattr(args, "risk_pct", 1.0))

    def _params(ticker: str):
        params = config.breakout(ticker) if ticker in config.breakout_names() else None
        if params is None:
            from bot_a.engine.breakout_fsm import BreakoutParams

            params = BreakoutParams()
        overrides = {}
        if getattr(args, "entry_mode", None):
            overrides["entry_mode"] = args.entry_mode
        if getattr(args, "overnight", None) is not None:
            overrides["allow_overnight_hold"] = bool(args.overnight)
            overrides["close_losers_before_clearing"] = bool(args.overnight)
        return params.overrides(**overrides) if overrides else params

    metas = {}
    for ticker in tickers:
        try:
            metas[ticker] = await asyncio.to_thread(fetch_futures_metadata, ticker)
        except Exception as exc:  # noqa: BLE001
            logger.error("Метаданные фьючерса %s недоступны: %s", ticker, exc)
            return 3
    logger.info(
        "Свинг-корзина %s: капитал %.0f ₽ (общий), риск %.2f %% на сделку, бары %d мин, "
        "утилизация ГО ≤ %.0f %%",
        ", ".join(tickers),
        capital,
        risk,
        bar_minutes,
        getattr(args, "margin_utilization", 60.0),
    )
    source = IssCandleSource(config.path(config.storage.data_cache_dir))

    if args.mode == "backtest":
        start_ms, end_ms = _parse_period(args)
        if end_ms is None:
            end_ms = int(datetime.now(tz=MSK).timestamp() * 1000)
        if start_ms is None:
            start_ms = end_ms - int(args.days) * 86_400_000
        data = {}
        for ticker, meta in metas.items():
            candles = await source.get_candles(meta, start_ms, end_ms, "1min")
            data[ticker] = _resample_if_needed(candles, bar_minutes)
        engine = SwingAccountBacktester(
            [(metas[ticker], _params(ticker)) for ticker in tickers],
            SwingAccountSettings(
                capital_rub=capital,
                risk_pct_per_trade=risk,
                tariff=config.fees.tariff,
                margin_safety_buffer_pct=config.risk.margin_safety_buffer_pct,
                max_margin_utilization_pct=float(getattr(args, "margin_utilization", 60.0)),
            ),
        )
        result = engine.run(data)
        curve = portfolio_curve(list(result.per_ticker.values()), capital)
        profits = sum(t.net_rub for t in result.trades if t.net_rub > 0)
        losses = -sum(t.net_rub for t in result.trades if t.net_rub <= 0)
        summary = {
            "mode": "ALL_SWING",
            "tickers": tickers,
            "barMinutes": bar_minutes,
            "capital": capital,
            "riskPct": risk,
            "trades": len(result.trades),
            "netRub": round(result.net_rub, 2),
            "feesRub": round(result.fees_rub, 2),
            "profitFactor": None if losses <= 0 else round(profits / losses, 3),
            "maxMarginUsed": round(result.max_margin_used, 0),
            "rejectedByMargin": result.rejected_by_margin,
            "rejectedByRisk": result.rejected_by_risk,
            "curve": curve,
            "perTicker": {
                ticker: block.summary(capital) for ticker, block in result.per_ticker.items()
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else _render_basket(summary))
        if not args.no_report:
            stem = args.report_stem or f"swing_basket_{datetime.now():%Y%m%d_%H%M%S}"
            path = config.path(config.storage.reports_dir) / f"{stem}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Отчёт корзины сохранён: %s", path)
        return 0

    if args.mode not in ("watch", "paper", "live"):
        logger.error("ALL_SWING поддерживает --mode backtest, watch, paper, live")
        return 2
    if args.mode == "live" and not args.yes:
        logger.error("Реальная торговля требует --yes")
        return 2

    warmup_days = max(int(getattr(args, "days", 10) or 10), 1)
    now_ms = int(datetime.now(tz=MSK).timestamp() * 1000)
    state_path = config.path(config.storage.state_dir) / "swing_positions.json"
    journal = config.path(config.storage.reports_dir) / (
        f"swing_basket_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    )
    runner = SwingBasketRunner(
        [(metas[ticker], _params(ticker)) for ticker in tickers],
        SwingBasketSettings(
            capital_rub=capital,
            risk_pct_per_trade=risk,
            margin_safety_buffer_pct=config.risk.margin_safety_pct
            if hasattr(config.risk, "margin_safety_pct")
            else config.risk.margin_safety_buffer_pct,
            max_margin_utilization_pct=float(getattr(args, "margin_utilization", 60.0)),
            bar_minutes=bar_minutes,
            poll_ms=int(args.record_interval_ms),
            state_path=state_path,
            journal_path=None if args.no_report else journal,
            tariff=config.fees.tariff,
            dry_run=args.mode != "live",
        ),
        logger=logger,
    )
    warmup: dict = {}
    for ticker, meta in metas.items():
        candles = await source.get_candles(meta, now_ms - warmup_days * 86_400_000, now_ms, "1min")
        bars = _resample_if_needed(candles, bar_minutes)
        warmup[ticker] = [
            Bar(
                time_ms=candle.time_ms,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=int(candle.volume),
            )
            for candle in bars
        ]
    runner.warmup(warmup)
    runner.restore()
    stop = asyncio.Event()
    _install_stop_handler(stop)
    token = config.t_invest.resolve_token(config.base_dir)
    clients: dict = {}
    if token:
        from bot_a.broker.tinkoff_rest_client import TInvestRestClient

        client = TInvestRestClient(
            token, config.t_invest.app_name, timeout=config.t_invest.request_timeout_seconds
        )
        clients = {ticker: client for ticker in tickers}
        logger.info("Свинг-корзина: стакан из T-Invest REST")
    else:
        clients = {
            ticker: IssQuoteSource(
                metas[ticker], depth=5, timeout=config.t_invest.request_timeout_seconds
            )
            for ticker in tickers
        }
        logger.info("Свинг-корзина: стакан из MOEX ISS (без токена)")
    summary = await runner.run(clients, duration_seconds=args.duration, stop_event=stop)
    runner.save_state()
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else _render_basket_live(summary))
    return 0


def _resample_if_needed(candles, bar_minutes: int):
    if bar_minutes > 1:
        return resample_candles(candles, bar_minutes)
    return candles


def _render_basket(summary: dict) -> str:
    pf = summary.get("profitFactor")
    curve = summary.get("curve", {})
    lines = [
        f"=== Свинг-корзина (бэктест): {', '.join(summary['tickers'])} ===",
        f"Бары {summary['barMinutes']} мин, капитал {summary['capital']:.0f} ₽ (общий), "
        f"риск {summary['riskPct']:.2f} % на сделку",
        f"Сделок: {summary['trades']}  Net: {summary['netRub']:+.2f} ₽  "
        f"комиссии: {summary['feesRub']:.2f} ₽",
        f"PF: {pf if pf is not None else 'н/д'}  Sharpe: {curve.get('sharpe')}  "
        f"Max DD: {curve.get('maxDrawdownRub', 0):.2f} ₽ ({curve.get('maxDrawdownPct', 0):.2f} %)",
        f"Пиковое ГО: {summary['maxMarginUsed']:.0f} ₽  "
        f"отказов по ГО: {summary['rejectedByMargin']}  по риску: {summary['rejectedByRisk']}",
    ]
    for ticker, block in sorted(summary["perTicker"].items()):
        lines.append(
            f"  {ticker:<9} сделок {block['trades']:>3} net {block['netRub']:>+9.2f} ₽ "
            f"PF {block['profitFactor']} WR {float(block['winRate']) * 100:.0f}%"
        )
    return "\n".join(lines)


def _render_basket_live(summary: dict) -> str:
    lines = [
        f"=== Свинг-корзина ({'watch' if summary['dryRun'] else 'paper/live'}) ===",
        f"Счёт: капитал {summary['capital']:.0f} ₽ → эквити {summary['equity']:.2f} ₽",
        f"Реализовано: {summary['realizedRub']:+.2f} ₽  комиссии: {summary['feesRub']:.2f} ₽  "
        f"сделок: {summary['trades']}  занято ГО: {summary['marginUsed']:.0f} ₽",
    ]
    for ticker, position in summary.get("openPositions", {}).items():
        lines.append(
            f"  {ticker}: {position['side']} {position['quantity']} по {position['entryPrice']} "
            f"(стоп {position['stopPrice']}, MFE {position['mfePoints']})"
        )
    return "\n".join(lines)


async def _dispatch_breakout(args: argparse.Namespace, config: AppConfig, logger) -> int:
    """`--strategy breakout` (v3.0): пробой сжатия волатильности на FORTS."""
    from bot_a.backtest.breakout_backtest import BreakoutBacktester, BreakoutSettings
    from bot_a.engine.breakout_fsm import Bar
    from bot_a.services.breakout_worker import BreakoutLiveWorker
    from bot_a.services.iss_quotes import IssQuoteSource
    from bot_a.utils.time_provider import MSK

    ticker = str(args.ticker or config.active_breakout or "").upper()
    if ticker == "ALL_SWING":
        return await _dispatch_swing_basket(args, config, logger)
    if not ticker:
        logger.error(
            "Не задан инструмент: укажите --ticker (например CNYRUBF) или activeBreakout в конфиге"
        )
        return 2
    params = config.breakout(ticker) if ticker in config.breakout_names() else None
    if params is None:
        from bot_a.engine.breakout_fsm import BreakoutParams

        params = BreakoutParams()
        logger.warning(
            "Для %s нет пресета в breakoutPresets — использую параметры по умолчанию", ticker
        )
    overrides = {}
    if getattr(args, "entry_mode", None):
        overrides["entry_mode"] = args.entry_mode
    if getattr(args, "overnight", None) is not None:
        overrides["allow_overnight_hold"] = bool(args.overnight)
        overrides["close_losers_before_clearing"] = bool(args.overnight)
    if overrides:
        params = params.overrides(**overrides)
    try:
        meta = await asyncio.to_thread(fetch_futures_metadata, ticker)
    except Exception as exc:  # noqa: BLE001 — метаданные FORTS обязательны
        logger.error("Метаданные фьючерса %s недоступны: %s", ticker, exc)
        return 3
    settings = BreakoutSettings(
        capital_rub=float(getattr(args, "capital", 50000.0)),
        risk_pct_per_trade=float(getattr(args, "risk_pct", 1.5)),
        tariff=config.fees.tariff,
        margin_safety_buffer_pct=config.risk.margin_safety_buffer_pct,
    )
    source = IssCandleSource(config.path(config.storage.data_cache_dir))
    bar_minutes = max(int(getattr(args, "bar_minutes", 1) or 1), 1)
    logger.info(
        "Пробой %s: шаг %s, 1 шаг = %s RUB, ГО %s RUB, капитал %.0f RUB, тариф %s, бары %d мин, "
        "вход %s, ночь %s",
        ticker,
        meta.min_price_increment,
        meta.min_price_increment_amount,
        meta.position_margin_rub(1),
        settings.capital_rub,
        config.fees.tariff.value,
        bar_minutes,
        params.entry_mode,
        "да" if params.allow_overnight_hold else "нет",
    )

    if args.mode == "backtest":
        start_ms, end_ms = _parse_period(args)
        if end_ms is None:
            end_ms = int(datetime.now(tz=MSK).timestamp() * 1000)
        if start_ms is None:
            start_ms = end_ms - int(args.days) * 86_400_000
        candles = await source.get_candles(meta, start_ms, end_ms, "1min")
        if not candles:
            logger.error("Нет свечей %s за запрошенный период", ticker)
            return 4
        if bar_minutes > 1:
            candles = resample_candles(candles, bar_minutes)
        result = BreakoutBacktester(meta, params, settings).run(candles)
        summary = result.summary(settings.capital_rub)
        summary["tradeList"] = [trade.to_dict() for trade in result.trades]
        logger.info(
            "Пробой %s: свечей %d, сделок %d, net %+.2f RUB, PF %s, WinRate %.1f%%",
            ticker,
            len(candles),
            summary["trades"],
            summary["netRub"],
            summary["profitFactor"],
            float(summary["winRate"]) * 100.0,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else _render_breakout(summary))
        if not args.no_report:
            stem = args.report_stem or f"breakout_{ticker}_{datetime.now():%Y%m%d_%H%M%S}"
            path = config.path(config.storage.reports_dir) / f"{stem}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Отчёт пробоя сохранён: %s", path)
        return 0

    if args.mode in ("watch", "paper"):
        warmup_days = max(int(getattr(args, "days", 5) or 5), 1)
        now_ms = int(datetime.now(tz=MSK).timestamp() * 1000)
        candles = await source.get_candles(meta, now_ms - warmup_days * 86_400_000, now_ms, "1min")
        if bar_minutes > 1:
            candles = resample_candles(candles, bar_minutes)
        journal = config.path(config.storage.reports_dir) / (
            f"breakout_{ticker}_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        )
        worker = BreakoutLiveWorker(
            meta,
            params,
            settings,
            dry_run=args.mode == "watch",
            poll_ms=int(args.record_interval_ms),
            journal_path=None if args.no_report else journal,
            logger=logger,
            tariff=config.fees.tariff,
            bar_minutes=bar_minutes,
        )
        worker.warmup(
            [
                Bar(
                    time_ms=candle.time_ms,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=int(candle.volume),
                )
                for candle in candles
            ]
        )
        stop = asyncio.Event()
        _install_stop_handler(stop)
        token = config.t_invest.resolve_token(config.base_dir)
        client: object
        if token:
            from bot_a.broker.tinkoff_rest_client import TInvestRestClient

            client = TInvestRestClient(
                token, config.t_invest.app_name, timeout=config.t_invest.request_timeout_seconds
            )
            logger.info("Пробой %s: стакан из T-Invest REST", ticker)
        else:
            client = IssQuoteSource(meta, depth=5, timeout=config.t_invest.request_timeout_seconds)
            logger.info("Пробой %s: стакан из MOEX ISS (без токена)", ticker)
        summary = await worker.run(
            client, duration_seconds=args.duration, stop_event=stop
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else _render_live(summary))
        return 0

    logger.error("--strategy breakout поддерживает --mode backtest, watch, paper")
    return 2


def _render_breakout(summary: dict) -> str:
    pf = summary["profitFactor"]
    lines = [
        f"=== Пробой v3.1: {summary['ticker']} ===",
        f"Баров: {summary['bars']}  Сделок: {summary['trades']} в {summary['days']} днях",
        f"Net PnL: {summary['netRub']:+.2f} RUB (gross {summary['grossRub']:+.2f}, "
        f"комиссии {summary['feesRub']:.2f})",
        f"WinRate: {float(summary['winRate']) * 100:.1f}%  PF: {pf if pf is not None else 'н/д'}  "
        f"Payoff: {summary['payoff']}",
        f"Средний плюс: {summary['avgWinner']:+.2f}  средний минус: {summary['avgLoser']:+.2f}",
        f"Max DD: {summary['maxDrawdownRub']:.2f} RUB ({summary['maxDrawdownPct']:.2f}%)",
        f"Доходность: {summary['returnPct']:+.2f}%  годовая: {summary['annualizedPct']:+.2f}%",
        f"Среднее удержание: {summary['avgHoldMinutes']:.1f} мин  "
        f"лучшая: {summary['bestTrade']:+.2f}  худшая: {summary['worstTrade']:+.2f}",
    ]
    return "\n".join(lines)


def _render_live(summary: dict) -> str:
    net = summary["net"]
    lines = [
        f"=== Пробой v3.1 (live): {summary['ticker']} ===",
        f"Баров: {summary['bars']}  Сделок: {net['trades']}",
        f"Реализовано: {net['realizedRub']:+.2f} RUB  Нереализовано: {net['unrealizedRub']:+.2f} RUB",
        f"Комиссии: {net['feesRub']:.2f} RUB  Итог: {summary['netRub']:+.2f} RUB",
    ]
    for trade in summary.get("tradeList", []):
        lines.append(
            f"  {trade['entry']} -> {trade['exit']} {trade['side']} {trade['quantity']} x "
            f"{trade['entryPrice']} ({trade['reason']}) net {trade['netRub']:+.2f}"
        )
    return "\n".join(lines)


async def _run_record(config: AppConfig, preset, args, logger, stop) -> dict:
    """Запись L2: gRPC SDK -> REST-поллинг -> ISS-фолбэк."""
    token = config.t_invest.resolve_token()
    if broker.sdk_available() and token:
        async with broker.async_client(
            token, config.t_invest.app_name, config.t_invest.request_timeout_seconds
        ) as client:
            leg1, leg2 = await broker.resolve_pair(client, preset.leg1, preset.leg2)
            resolved = preset.with_metadata(leg1, leg2)
            logger.info(
                "Запись через gRPC: %s uid=%s / %s uid=%s",
                leg1.ticker, leg1.instrument_uid, leg2.ticker, leg2.instrument_uid,
            )
            service = RecordingService(config, resolved, client, logger)
            return await service.run(duration_seconds=args.duration, stop_event=stop)

    if token:
        from bot_a.broker.tinkoff_rest_client import RestRecordingService, TInvestRestClient

        client = TInvestRestClient(
            token,
            config.t_invest.app_name,
            timeout=config.t_invest.request_timeout_seconds,
        )
        leg1 = await client.resolve_instrument(preset.leg1.ticker, preset.leg1.class_code, preset.leg1.market)
        leg2 = await client.resolve_instrument(preset.leg2.ticker, preset.leg2.class_code, preset.leg2.market)
        resolved = preset.with_metadata(leg1, leg2)
        logger.info(
            "Запись через REST: %s uid=%s / %s uid=%s, интервал %d мс, глубина %d",
            leg1.ticker, leg1.instrument_uid, leg2.ticker, leg2.instrument_uid,
            args.record_interval_ms, args.record_depth,
        )
        service = RestRecordingService(
            config,
            resolved,
            client,
            interval_ms=args.record_interval_ms,
            depth=args.record_depth,
            logger=logger,
        )
        return await service.run(duration_seconds=args.duration, stop_event=stop)

    from bot_a.backtest.iss_source import IssRecordingService

    logger.warning(
        "Токен T-Invest не задан: запись через публичный MOEX ISS (снимки стакана, интервал >=1000 мс)"
    )
    service = IssRecordingService(
        config,
        preset,
        interval_ms=max(int(args.record_interval_ms), 1000),
        depth=args.record_depth,
        logger=logger,
    )
    return await service.run(duration_seconds=args.duration, stop_event=stop)


async def _dispatch(args: argparse.Namespace, config: AppConfig, logger) -> int:
    if (args.strategy or config.strategy_mode) == "breakout":
        return await _dispatch_breakout(args, config, logger)
    if (args.preset or "").upper() == "ALL_ACTIVE" and args.mode in ("watch", "paper"):
        return await _run_all_active(args, config, logger)
    preset = config.preset()
    logger.info(
        "Bot A: режим=%s, пара=%s (%s/%s), тариф=%s, топология=%s",
        args.mode,
        preset.name,
        preset.leg1.ticker,
        preset.leg2.ticker,
        config.fees.tariff.value,
        config.execution.topology.value,
    )
    start_ms, end_ms = _parse_period(args)
    stop = asyncio.Event()
    _install_stop_handler(stop)
    token = config.t_invest.resolve_token(config.base_dir)

    if args.mode == "preflight":
        readiness = None
        if getattr(args, "monday_ready", False):
            from bot_a.services.readiness import build_monday_readiness

            readiness = build_monday_readiness(config, logger=logger)
            print(readiness.render())
        try:
            if not broker.sdk_available() and token:
                # Python 3.14: gRPC SDK недоступен — проверяем через REST/JSON.
                from bot_a.broker.tinkoff_rest_client import TInvestRestClient, TInvestRestError
                from bot_a.services.preflight import run_preflight_rest

                logger.info("Preflight через REST/JSON T-Invest (gRPC SDK недоступен)")
                try:
                    client = TInvestRestClient(
                        token,
                        config.t_invest.app_name,
                        timeout=config.t_invest.request_timeout_seconds,
                    )
                    report = await run_preflight_rest(config, preset, client, logger)
                except TInvestRestError as exc:
                    logger.error(
                        "REST preflight не удался: %s (источник токена: %s)",
                        exc,
                        config.t_invest.token_source(config.base_dir),
                    )
                    return 3
            else:
                report = await run_preflight(config, preset, logger)
        except Exception as exc:  # нет токена/SDK — офлайн-проверка уже показана
            if readiness is None:
                raise
            logger.error("Preflight через T-Invest недоступен: %s", exc)
            if args.json:
                print(json.dumps(readiness.to_dict(), ensure_ascii=False, indent=2))
            return 2
        print(report.render())
        if args.json:
            payload = report.to_dict()
            if readiness is not None:
                payload["mondayReady"] = readiness.to_dict()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if report.ok else 2

    if args.mode == "backtest":
        from bot_a.backtest.charts import TraceRecorder, render_charts

        trace = TraceRecorder() if args.charts else None
        charts: list = []
        candles1 = candles2 = ()
        if args.source == "recordings":
            source = build_replay_source(config, args.recordings)
            backtester = TickReplayBacktester(config, preset, source, logger=logger)
            replay_range = ReplayRange(
                start_ms if start_ms is not None else 0,
                end_ms if end_ms is not None else ReplayRange.full().end_ms,
            )
            result = await backtester.run(
                replay_range,
                write_report=not args.no_report,
                report_stem=args.report_stem,
                trace=trace,
            )
        elif args.source == "iss":
            resolved = await _resolve_iss_preset(preset)
            logger.info(
                "MOEX ISS: %s шаг=%s (1 шаг = %s RUB, ГО %s) / %s шаг=%s (1 шаг = %s RUB, ГО %s)",
                resolved.leg1.ticker,
                resolved.leg1.min_price_increment,
                resolved.leg1.min_price_increment_amount,
                resolved.leg1.position_margin_rub(1),
                resolved.leg2.ticker,
                resolved.leg2.min_price_increment,
                resolved.leg2.min_price_increment_amount,
                resolved.leg2.position_margin_rub(1),
            )
            source = IssCandleSource(config.path(config.storage.data_cache_dir))
            backtester = HistoricalCandleBacktester(
                config, resolved, source, logger=logger, synth_mode=args.synth_mode
            )
            result = await backtester.run(
                days=args.days,
                start_ms=start_ms,
                end_ms=end_ms,
                interval=args.interval,
                write_report=not args.no_report,
                report_stem=args.report_stem,
                trace=trace,
            )
            candles1, candles2 = backtester.last_candles
        else:
            async with broker.async_client(
                token, config.t_invest.app_name, config.t_invest.request_timeout_seconds
            ) as client:
                leg1, leg2 = await broker.resolve_pair(client, preset.leg1, preset.leg2)
                resolved = preset.with_metadata(leg1, leg2)
                logger.info(
                    "Инструменты разрешены: %s uid=%s, %s uid=%s",
                    leg1.ticker, leg1.instrument_uid, leg2.ticker, leg2.instrument_uid,
                )
                source = TInvestCandleSource(client, config.path(config.storage.data_cache_dir), args.interval)
                backtester = HistoricalCandleBacktester(
                    config, resolved, source, logger=logger, synth_mode=args.synth_mode
                )
                result = await backtester.run(
                    days=args.days,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    interval=args.interval,
                    write_report=not args.no_report,
                    report_stem=args.report_stem,
                    trace=trace,
                )
                candles1, candles2 = backtester.last_candles
        if args.charts:
            stem = args.report_stem or f"charts_{preset.name}_{datetime.now():%Y%m%d_%H%M%S}"
            chart_dir = args.chart_dir or str(
                config.path(config.storage.reports_dir) / "charts" / stem
            )
            charts = render_charts(
                result,
                trace,
                chart_dir,
                stem,
                preset_label=f"{preset.leg1.ticker}/{preset.leg2.ticker}",
                candles1=candles1,
                candles2=candles2,
                zoom_trades=max(int(args.chart_trades), 0),
                logger=logger,
            )
            for chart in charts:
                print(f"график: {chart}", file=sys.stderr)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.mode == "optimize":
        if args.source == "iss":
            resolved = await _resolve_iss_preset(preset)
            source = IssCandleSource(config.path(config.storage.data_cache_dir))
            optimizer = GridSearchOptimizer(
                config, resolved, source, base_params=config.strategy, jobs=args.jobs, logger=logger
            )
            results = await optimizer.run(
                days=args.days,
                start_ms=start_ms,
                end_ms=end_ms,
                interval=args.interval,
                write_reports=not args.no_report,
            )
        else:
            async with broker.async_client(
                token, config.t_invest.app_name, config.t_invest.request_timeout_seconds
            ) as client:
                leg1, leg2 = await broker.resolve_pair(client, preset.leg1, preset.leg2)
                resolved = preset.with_metadata(leg1, leg2)
                source = TInvestCandleSource(client, config.path(config.storage.data_cache_dir), args.interval)
                optimizer = GridSearchOptimizer(
                    config, resolved, source, base_params=config.strategy, jobs=args.jobs, logger=logger
                )
                results = await optimizer.run(
                    days=args.days,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    interval=args.interval,
                    write_reports=not args.no_report,
                )
        print(optimizer.render_table(results, limit=12))
        if args.json and results:
            print(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))
        return 0

    if args.mode == "record":
        stats = await _run_record(config, preset, args, logger, stop)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    if args.mode == "watch":
        # Живой стакан + ядро бота + дашборд в браузере (реальные данные, без денег).
        from bot_a.broker.tinkoff_rest_client import TInvestRestClient, TInvestRestError
        from bot_a.services.live_dashboard import LiveWatchService

        try:
            client = TInvestRestClient(
                token,
                config.t_invest.app_name,
                timeout=config.t_invest.request_timeout_seconds,
            )
            leg1 = await client.resolve_instrument(
                preset.leg1.ticker, preset.leg1.class_code, preset.leg1.market
            )
            leg2 = await client.resolve_instrument(
                preset.leg2.ticker, preset.leg2.class_code, preset.leg2.market
            )
        except TInvestRestError as exc:
            logger.error(
                "T-Invest REST недоступен: %s (источник токена: %s)",
                exc,
                config.t_invest.token_source(config.base_dir),
            )
            return 3
        resolved = preset.with_metadata(leg1, leg2)
        service = LiveWatchService(
            config,
            resolved,
            client,
            interval_ms=int(args.record_interval_ms),
            depth=int(args.record_depth),
            port=int(args.port),
            logger=logger,
        )
        print(f"Дашборд: http://127.0.0.1:{int(args.port)}/  (Ctrl+C — остановка)")
        await service.run(duration_seconds=args.duration, stop_event=stop)
        return 0

    if args.mode in ("paper", "live"):
        if args.mode == "live" and not args.yes:
            logger.error(
                "Режим live требует подтверждения: добавьте --yes. "
                "Сначала прогоните --mode paper на тех же параметрах."
            )
            return 2
        if args.source == "iss":
            if args.mode == "live":
                logger.error("Источник iss доступен только для --mode paper (реальные заявки — только через API)")
                return 2
            from bot_a.backtest.charts import TraceRecorder, render_charts
            from bot_a.services.paper_replay import PaperReplayService

            resolved = await _resolve_iss_preset(preset)
            service = PaperReplayService(
                config, resolved, logger=logger, synth_mode=args.synth_mode
            )
            if start_ms is None or end_ms is None:
                day = datetime.now(MSK).date()
                start_ms = msk_to_ms(datetime.combine(day, time(9, 50, tzinfo=MSK)))
                end_ms = msk_to_ms(datetime.combine(day, time(23, 50, tzinfo=MSK)))
            trace = TraceRecorder()
            result, warmup, (candles1, candles2) = await service.run(
                start_ms,
                end_ms,
                warmup_days=int(config.execution.bootstrap_warmup_days),
                trace=trace,
            )
            stem = args.report_stem or f"paper_{preset.name}_{datetime.now():%Y%m%d_%H%M%S}"
            chart_dir = args.chart_dir or str(config.path(config.storage.reports_dir) / "charts" / stem)
            charts = render_charts(
                result,
                trace,
                chart_dir,
                stem,
                preset_label=f"{preset.leg1.ticker}/{preset.leg2.ticker}",
                candles1=candles1,
                candles2=candles2,
                zoom_trades=max(int(args.chart_trades), 0),
                logger=logger,
            )
            totals = result.totals
            print(
                f"PAPER (реальные данные MOEX ISS) {preset.name}: "
                f"сделок {totals.trades}, gross {totals.gross_pnl_rub} RUB, "
                f"комиссии {totals.fees_rub} RUB, NET {totals.net_pnl_rub} RUB, "
                f"PF {totals.profit_factor:.2f}, дней в плюсе {totals.profitable_days}/{totals.total_days}"
            )
            for chart in charts:
                print(f"график: {chart}", file=sys.stderr)
            if args.json:
                payload = result.to_dict()
                payload["warmup"] = warmup
                payload["charts"] = [str(chart) for chart in charts]
                payload["mode"] = "paper-iss"
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.mode == "paper" and not broker.sdk_available():
            # Python 3.14: gRPC-стрима нет, но REST-поллинг стакана доступен.
            from bot_a.backtest.charts import TraceRecorder, render_charts
            from bot_a.broker.tinkoff_rest_client import TInvestRestClient, TInvestRestError
            from bot_a.services.rest_paper import RestPaperService

            client = None
            try:
                client = TInvestRestClient(
                    token,
                    config.t_invest.app_name,
                    timeout=config.t_invest.request_timeout_seconds,
                )
                leg1 = await client.resolve_instrument(
                    preset.leg1.ticker, preset.leg1.class_code, preset.leg1.market
                )
                leg2 = await client.resolve_instrument(
                    preset.leg2.ticker, preset.leg2.class_code, preset.leg2.market
                )
            except TInvestRestError as exc:
                logger.error(
                    "T-Invest REST недоступен: %s. Проверьте токен (источник: %s) и доступ к "
                    "invest-public-api.tinkoff.ru",
                    exc,
                    config.t_invest.token_source(config.base_dir),
                )
                return 3
            resolved = preset.with_metadata(leg1, leg2)
            logger.info(
                "Инструменты разрешены через REST: %s uid=%s, %s uid=%s",
                leg1.ticker, leg1.instrument_uid, leg2.ticker, leg2.instrument_uid,
            )
            service = RestPaperService(
                config,
                resolved,
                client,
                interval_ms=int(args.record_interval_ms),
                depth=int(args.record_depth),
                logger=logger,
            )
            trace = TraceRecorder()
            snapshot, trades, warmup = await service.run(
                duration_seconds=args.duration,
                stop_event=stop,
                trace=trace,
            )
            result = _result_from_snapshot(config, resolved, snapshot, trades)
            stem = args.report_stem or f"paper_live_{preset.name}_{datetime.now():%Y%m%d_%H%M%S}"
            chart_dir = args.chart_dir or str(config.path(config.storage.reports_dir) / "charts" / stem)
            charts = render_charts(
                result,
                trace,
                chart_dir,
                stem,
                preset_label=f"{preset.leg1.ticker}/{preset.leg2.ticker} (live L2)",
                zoom_trades=0,
                logger=logger,
            )
            totals = result.totals
            print(
                f"PAPER (живой стакан T-Invest через REST) {preset.name}: "
                f"сделок {totals.trades}, gross {totals.gross_pnl_rub} RUB, "
                f"комиссии {totals.fees_rub} RUB, NET {totals.net_pnl_rub} RUB, "
                f"стаканов {snapshot.get('books')}, сессия {snapshot.get('sessionSeconds')} c"
            )
            for chart in charts:
                print(f"график: {chart}", file=sys.stderr)
            if args.json:
                payload = result.to_dict()
                payload["warmup"] = warmup
                payload["snapshot"] = snapshot
                payload["charts"] = [str(chart) for chart in charts]
                payload["mode"] = "paper-rest"
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        async with broker.async_client(
            token, config.t_invest.app_name, config.t_invest.request_timeout_seconds
        ) as client:
            worker = LiveTradingWorker(config, preset, client, paper=args.mode == "paper", logger=logger)
            snapshot = await worker.run(stop_event=stop, duration_seconds=args.duration)
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
        return 0

    raise ValueError(f"Неизвестный режим: {args.mode}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = AppConfig.load(args.config)
    except FileNotFoundError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 4
    except json.JSONDecodeError as exc:
        print(f"Ошибка JSON в конфигурации: {exc}", file=sys.stderr)
        return 4

    config = _apply_cli_overrides(config, args)
    all_active = (args.preset or "").upper() == "ALL_ACTIVE"
    if not all_active:
        try:
            preset = config.preset()
        except KeyError as exc:
            print(f"Ошибка пресета: {exc}", file=sys.stderr)
            return 4
        if not preset.enabled and args.mode in ("paper", "live"):
            print(
                f"Пресет '{preset.name}' отключён от торговли (enabled = false). "
                f"Активные пары: {', '.join(config.trading_preset_names()) or '—'}",
                file=sys.stderr,
            )
            return 4
    config.ensure_directories()
    logger = configure_logging(
        config.logging.level,
        str(config.path(config.logging.log_dir)),
        config.logging.file_name,
        console_stream=sys.stderr if args.json else None,
    )

    strategy_breakout = (args.strategy or config.strategy_mode) == "breakout"
    needs_tinvest = (
        args.mode in ("live", "watch")
        or (args.mode == "paper" and args.source != "iss")
        or (args.mode == "preflight" and not getattr(args, "monday_ready", False))
        or (args.mode == "backtest" and args.source == "api")
        or (args.mode == "optimize" and args.source == "api")
    )
    if strategy_breakout and args.mode in ("backtest", "watch", "paper"):
        # v3.0: история и стакан FORTS доступны через MOEX ISS — токен не обязателен.
        needs_tinvest = False
    if needs_tinvest:
        if not config.t_invest.resolve_token(config.base_dir):
            logger.error(
                "Не задан токен T-Invest. Варианты: переменная %s, файл secrets/tinvest_token.txt "
                "(или TINVEST_TOKEN_FILE), поле tInvest.token в appsettings.json, флаг --token",
                config.t_invest.token_env_var,
            )
            return 3
        rest_covers = args.mode in ("paper", "record", "preflight", "watch")
        if not broker.sdk_available() and not rest_covers:
            logger.error(broker.SDK_INSTALL_HINT)
            return 3
        if not broker.sdk_available() and rest_covers:
            logger.warning(
                "gRPC SDK недоступен: режим пойдёт через REST/JSON T-Invest "
                "(GetOrderBook %d мс, глубина %d)",
                args.record_interval_ms,
                args.record_depth,
            )

    try:
        return asyncio.run(_dispatch(args, config, logger))
    except broker.SdkUnavailableError as exc:
        logger.error("%s", exc)
        return 3
    except KeyboardInterrupt:
        logger.warning("Прервано пользователем")
        return 130
    except Exception as exc:  # верхнеуровневая защита CLI
        logger.exception("Необработанная ошибка: %s", exc)
        return 1
