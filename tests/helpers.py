"""Общие фабрики для тестов."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from bot_a.backtest.candles import Candle
from bot_a.config import (
    AppConfig,
    ExecutionSettings,
    FeeSettings,
    PairPreset,
    RiskSettings,
    StorageSettings,
    TInvestSettings,
)
from bot_a.engine.fsm import StrategyParams
from bot_a.models.domain import (
    ExecutionTopology,
    InstrumentMetadata,
    MarketKind,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
    TradePrint,
)
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import MSK, msk_to_ms


def make_meta(
    ticker: str = "AAA",
    class_code: str = "SPBFUT",
    price_step: str = "0.01",
    step_amount: str = "0.01",
    margin: str = "1000",
    futures: bool = True,
    uid: Optional[str] = None,
) -> InstrumentMetadata:
    return InstrumentMetadata(
        instrument_uid=uid or f"uid-{ticker}",
        ticker=ticker,
        class_code=class_code,
        market=MarketKind.FUTURES if futures else MarketKind.SPOT,
        lot_size=1,
        min_price_increment=Decimal(price_step),
        min_price_increment_amount=Decimal(step_amount),
        initial_margin_buy=Decimal(margin),
        initial_margin_sell=Decimal(margin),
    )


def make_spot(
    ticker: str = "AAA",
    lot: int = 1,
    price_step: str = "0.01",
    uid: Optional[str] = None,
) -> InstrumentMetadata:
    return InstrumentMetadata(
        instrument_uid=uid or f"uid-{ticker}",
        ticker=ticker,
        class_code="TQBR",
        market=MarketKind.SPOT,
        lot_size=lot,
        min_price_increment=Decimal(price_step),
    )


def make_preset(
    leg1: Optional[InstrumentMetadata] = None,
    leg2: Optional[InstrumentMetadata] = None,
    name: str = "TEST_PAIR",
    beta: float = 1.0,
) -> PairPreset:
    return PairPreset(
        name=name,
        description="тестовая пара",
        leg1=leg1 or make_meta("AAA"),
        leg2=leg2 or make_meta("BBB"),
        beta=beta,
    )


def make_config(
    tmp_path,
    preset: Optional[PairPreset] = None,
    params: Optional[StrategyParams] = None,
    topology: ExecutionTopology = ExecutionTopology.SINGLE_LEG_LEAD_LAG,
    tariff=None,
) -> AppConfig:
    from bot_a.economics.fees import Tariff

    pair = preset or make_preset()
    strategy = params or StrategyParams(
        z_entry=1.0,
        z_exit=0.2,
        z_cancel=0.5,
        ewma_window=20,
        min_edge_multiplier=Decimal("1.0"),
        slippage_buffer_steps=0,
        max_hold_seconds=180,
        max_exit_quote_seconds=5,
        cooldown_seconds=1,
        requote_cooldown_ms=0,
        quantity_lots=1,
    )
    return AppConfig(
        t_invest=TInvestSettings(token="test-token"),
        fees=FeeSettings(tariff=tariff or Tariff.TRADER),
        risk=RiskSettings(
            max_daily_drawdown_rub=Decimal("100000"),
            margin_safety_buffer_pct=25.0,
            trade_evening_session=True,
        ),
        execution=ExecutionSettings(
            topology=topology,
            simulated_order_latency_ms=0,
            queue_fill_factor=0.35,
            slippage_steps=0,
            order_book_depth=5,
            paper_initial_cash_rub=Decimal("1000000"),
        ),
        storage=StorageSettings(
            postgres_dsn="",
            recordings_dir=str(tmp_path / "recordings"),
            data_cache_dir=str(tmp_path / "cache"),
            reports_dir=str(tmp_path / "reports"),
            record_to_postgres=False,
            record_to_files=True,
        ),
        strategy=strategy,
        presets={pair.name: pair},
        active_preset=pair.name,
        base_dir=str(tmp_path),
    )


def make_book(
    mid: str | float | Decimal,
    time_ms: int,
    depth: int = 5,
    obi: float = 0.0,
    uid: str = "",
    step: str = "0.01",
    volume: int = 100,
) -> OrderBookSnapshot:
    increment = Decimal(step)
    best_bid = dec(mid).quantize(increment)
    best_ask = best_bid + increment
    bid_volume = max(int(volume * (1.0 + obi) / 2.0), 1)
    ask_volume = max(int(volume * (1.0 - obi) / 2.0), 1)
    bids = tuple(
        OrderBookLevel(price=best_bid - increment * index, quantity=max(bid_volume // depth, 1))
        for index in range(depth)
    )
    asks = tuple(
        OrderBookLevel(price=best_ask + increment * index, quantity=max(ask_volume // depth, 1))
        for index in range(depth)
    )
    return OrderBookSnapshot(instrument_uid=uid, time_ms=time_ms, bids=bids, asks=asks, last_price=best_bid)


def make_trade(price: str | Decimal, time_ms: int, quantity: int = 50, uid: str = "", side: Side = Side.BUY) -> TradePrint:
    return TradePrint(instrument_uid=uid, time_ms=time_ms, price=dec(price), quantity=quantity, side=side)




def make_candles(
    start_ms: int,
    count: int,
    start_price: str = "100",
    drift_per_minute: str = "0",
    volume: int = 1000,
    amplitude: str = "0.05",
) -> List[Candle]:
    """Детерминированные минутные свечи с пилообразным движением."""
    price = dec(start_price)
    drift = dec(drift_per_minute)
    amp = dec(amplitude)
    candles: List[Candle] = []
    for index in range(count):
        open_price = price
        close_price = open_price + drift + (amp if index % 2 == 0 else -amp)
        high = max(open_price, close_price) + amp / 2
        low = min(open_price, close_price) - amp / 2
        candles.append(
            Candle(
                time_ms=start_ms + index * 60_000,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=volume,
            )
        )
        price = close_price
    return candles


def trading_start_ms(day: str = "2025-03-11", hour: int = 10, minute: int = 0) -> int:
    date = datetime.fromisoformat(day).date()
    return msk_to_ms(datetime(date.year, date.month, date.day, hour, minute, tzinfo=MSK))
