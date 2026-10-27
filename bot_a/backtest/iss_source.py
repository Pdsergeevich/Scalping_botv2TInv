"""Источник свечей MOEX ISS (публичный, без токена).

Используется как исследовательский резерв, когда недоступен T-Invest SDK:
те же реальные минутные свечи MOEX, та же логика бэктеста. Для торговли
используется только T-Invest (`--source api`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from bot_a.backtest.candles import Candle
from bot_a.models.domain import InstrumentMetadata, MarketKind, OrderBookLevel, OrderBookSnapshot, Side, TradePrint
from bot_a.utils.quotation import dec
from bot_a.utils.time_provider import MSK

ISS_BASE = "https://iss.moex.com/iss"
_USER_AGENT = {"User-Agent": "BotA-Research/1.0"}
_PAGE_SIZE = 500


def _get_json(url: str, timeout: int = 60) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers=_USER_AGENT)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    if not raw.lstrip().startswith(("{", "[")):
        raise RuntimeError(
            "MOEX ISS вернул не JSON (вероятно, ресурс доступен только по подписке)")
    return json.loads(raw)


def fetch_futures_metadata(ticker: str) -> InstrumentMetadata:
    """Метаданные фьючерса FORTS: шаг, стоимость шага, ГО, размер контракта."""
    payload = _get_json(
        f"{ISS_BASE}/engines/futures/markets/forts/securities/{ticker}.json?iss.meta=off"
    )["securities"]
    row = dict(zip(payload["columns"], payload["data"][0]))
    return InstrumentMetadata(
        instrument_uid=ticker,
        ticker=ticker,
        class_code="SPBFUT",
        market=MarketKind.FUTURES,
        name=str(row.get("SHORTNAME") or ticker),
        lot_size=int(row.get("LOTVOLUME") or 1),
        min_price_increment=dec(row.get("MINSTEP") or "0.01"),
        min_price_increment_amount=dec(row.get("STEPPRICE") or "0"),
        initial_margin_buy=dec(row.get("INITIALMARGIN") or 0),
        initial_margin_sell=dec(row.get("INITIALMARGIN") or 0),
    )


def fetch_stock_metadata(ticker: str, board: str = "TQBR") -> InstrumentMetadata:
    """Метаданные акции: размер лота из списка бумаг режима ``board``."""
    payload = _get_json(
        f"{ISS_BASE}/engines/stock/markets/shares/boards/{board}/securities.json"
        f"?iss.meta=off&iss.only=securities"
        f"&securities.columns=SECID,SHORTNAME,LOTSIZE,MINSTEP"
    )
    columns = payload["securities"]["columns"]
    index = {name: position for position, name in enumerate(columns)}
    for row in payload["securities"]["data"]:
        if str(row[index["SECID"]]).upper() != ticker.upper():
            continue
        return InstrumentMetadata(
            instrument_uid=ticker,
            ticker=ticker,
            class_code=board,
            market=MarketKind.SPOT,
            name=str(row[index["SHORTNAME"]] or ticker),
            lot_size=int(row[index["LOTSIZE"]] or 1),
            min_price_increment=dec(row[index["MINSTEP"]] or "0.01"),
        )
    raise RuntimeError(f"Инструмент {ticker} не найден в режиме {board}")


def fetch_pair(
    ticker1: str,
    ticker2: str,
    futures: bool = True,
) -> tuple[InstrumentMetadata, InstrumentMetadata]:
    loader = fetch_futures_metadata if futures else fetch_stock_metadata
    return loader(ticker1), loader(ticker2)


class IssCandleSource:
    """``CandleSource`` поверх MOEX ISS с дневным кэшем на диске."""

    def __init__(self, cache_dir: str | Path, timeout: int = 60, today_ttl_seconds: float = 120.0) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = int(timeout)
        self._today_ttl = float(today_ttl_seconds)

    def _cache_path(self, meta: InstrumentMetadata, day) -> Path:
        market = "forts" if meta.is_futures else "tqbr"
        return self._cache_dir / f"iss_{market}_{meta.ticker}_{day.strftime('%Y%m%d')}.json"

    def _cache_is_fresh(self, path: Path, day) -> bool:
        """Прошедшие дни в кэше неизменяемы, текущий день дочитывается по мере торгов."""
        if not path.exists():
            return False
        if day != datetime.now(tz=MSK).date():
            return True
        return (time.time() - path.stat().st_mtime) <= self._today_ttl

    def _url(self, meta: InstrumentMetadata, date_from: str, date_to: str, start: int) -> str:
        if meta.is_futures:
            base = f"{ISS_BASE}/engines/futures/markets/forts/securities/{meta.ticker}/candles.json"
        else:
            base = (
                f"{ISS_BASE}/engines/stock/markets/shares/boards/{meta.class_code}"
                f"/securities/{meta.ticker}/candles.json"
            )
        return f"{base}?interval=1&from={date_from}&till={date_to}&start={start}&iss.meta=off"

    def _fetch_day(self, meta: InstrumentMetadata, day) -> List[Dict[str, Any]]:
        date_from = day.strftime("%Y-%m-%d")
        candles: List[Dict[str, Any]] = []
        start = 0
        while True:
            payload = _get_json(self._url(meta, date_from, date_from, start), self._timeout)["candles"]
            columns = payload["columns"]
            rows = payload["data"]
            if not rows:
                break
            index = {name: position for position, name in enumerate(columns)}
            for row in rows:
                begin = datetime.fromisoformat(row[index["begin"]]).replace(tzinfo=timezone.utc)
                candles.append(
                    {
                        "time": int(begin.timestamp() * 1000),
                        "open": str(row[index["open"]]),
                        "high": str(row[index["high"]]),
                        "low": str(row[index["low"]]),
                        "close": str(row[index["close"]]),
                        "volume": int(row[index["volume"]] or 0),
                    }
                )
            start += len(rows)
            if len(rows) < _PAGE_SIZE:
                break
        candles.sort(key=lambda item: item["time"])
        return candles

    async def get_candles(
        self,
        meta: InstrumentMetadata,
        start_ms: int,
        end_ms: int,
        interval: str = "1min",
    ) -> List[Candle]:
        first_day = datetime.fromtimestamp(start_ms / 1000, tz=MSK).date()
        last_day = datetime.fromtimestamp(end_ms / 1000, tz=MSK).date()
        candles: List[Candle] = []
        day = first_day
        while day <= last_day:
            path = self._cache_path(meta, day)
            if self._cache_is_fresh(path, day):
                raw = json.loads(path.read_text(encoding="utf-8"))
            else:
                fresh = await asyncio.to_thread(self._fetch_day, meta, day)
                raw = fresh
                if path.exists():
                    try:
                        old = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        old = []
                    merged = {item["time"]: item for item in old if "time" in item}
                    merged.update({item["time"]: item for item in fresh})
                    raw = [merged[key] for key in sorted(merged)]
                path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            candles.extend(Candle.from_dict(item) for item in raw)
            day += timedelta(days=1)
        result = [item for item in candles if start_ms <= item.time_ms <= end_ms]
        result.sort(key=lambda item: item.time_ms)
        unique: Dict[int, Candle] = {item.time_ms: item for item in result}
        return [unique[key] for key in sorted(unique)]


def pair_preset(
    name: str,
    ticker1: str,
    ticker2: str,
    futures: bool = True,
    beta: float = 1.0,
    description: str = "MOEX ISS research pair",
):
    """Собирает ``PairPreset`` с реальными метаданными ISS."""
    from bot_a.config import PairPreset

    leg1, leg2 = fetch_pair(ticker1, ticker2, futures)
    return PairPreset(name=name, description=description, leg1=leg1, leg2=leg2, beta=beta)

class IssRecordingService:
    """Fallback-запись L2 без токена: опрос стаканов и сделок через MOEX ISS."""

    def __init__(
        self,
        config: Any,
        preset: Any,
        interval_ms: int = 1000,
        depth: int = 20,
        logger: Any = None,
    ) -> None:
        self._config = config
        self._preset = preset
        self._interval = max(int(interval_ms), 250)
        self._depth = max(int(depth), 1)
        self._log = logger or logging.getLogger("bot_a.recorder.iss")
        self._seen_trades: set = set()

    def _book_url(self, meta: InstrumentMetadata) -> str:
        if meta.is_futures:
            base = f"{ISS_BASE}/engines/futures/markets/forts/securities/{meta.ticker}"
        else:
            base = (
                f"{ISS_BASE}/engines/stock/markets/shares/boards/{meta.class_code}"
                f"/securities/{meta.ticker}"
            )
        return f"{base}/orderbook.json?depth={self._depth}"

    def _trades_url(self, meta: InstrumentMetadata) -> str:
        if meta.is_futures:
            base = f"{ISS_BASE}/engines/futures/markets/forts/securities/{meta.ticker}"
        else:
            base = (
                f"{ISS_BASE}/engines/stock/markets/shares/boards/{meta.class_code}"
                f"/securities/{meta.ticker}"
            )
        return f"{base}/trades.json"

    @staticmethod
    def _rows(payload: Dict[str, Any], key: str) -> list:
        block = payload.get(key) or {}
        columns = block.get("columns") or []
        index = {name: position for position, name in enumerate(columns)}
        rows = []
        for row in block.get("data") or []:
            rows.append({name: row[position] for name, position in index.items()})
        return rows

    def _parse_book(self, meta: InstrumentMetadata, payload: Dict[str, Any]) -> Optional[OrderBookSnapshot]:
        rows = self._rows(payload, "orderbook")
        if not rows:
            return None
        bids = []
        asks = []
        time_ms = 0
        for row in rows:
            side = str(row.get("BUYSELL") or "").upper()
            price = dec(row.get("PRICE") if row.get("PRICE") is not None else 0)
            quantity = int(row.get("QUANTITY") or 0)
            if price <= 0 or quantity <= 0:
                continue
            level = (price, quantity)
            if side == "B":
                bids.append(level)
            elif side == "S":
                asks.append(level)
            stamp = row.get("UPDATETIME")
            if stamp:
                time_ms = self._msk_time_to_ms(str(stamp))
        if not bids and not asks:
            return None
        bids.sort(key=lambda item: item[0], reverse=True)
        asks.sort(key=lambda item: item[0])
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        return OrderBookSnapshot(
            instrument_uid=meta.instrument_uid or meta.ticker,
            time_ms=time_ms or now_ms,
            bids=tuple(OrderBookLevel(price=price, quantity=quantity) for price, quantity in bids[: self._depth]),
            asks=tuple(OrderBookLevel(price=price, quantity=quantity) for price, quantity in asks[: self._depth]),
        )

    @staticmethod
    def _msk_time_to_ms(stamp: str) -> int:
        try:
            moment = datetime.now(tz=timezone.utc)
            hour, minute, second = (int(part) for part in stamp.split(":")[:3])
            return int(
                moment.replace(hour=hour, minute=minute, second=second, microsecond=0).timestamp() * 1000
            )
        except (ValueError, TypeError):
            return 0

    def _parse_trades(self, meta: InstrumentMetadata, payload: Dict[str, Any]) -> List[TradePrint]:
        result: List[TradePrint] = []
        for row in self._rows(payload, "trades"):
            trade_no = row.get("TRADENO")
            key = (meta.ticker, row.get("TRADEDATE"), trade_no)
            if trade_no is not None:
                if key in self._seen_trades:
                    continue
                self._seen_trades.add(key)
            price = dec(row.get("PRICE") or 0)
            quantity = int(row.get("QUANTITY") or 0)
            if price <= 0 or quantity <= 0:
                continue
            stamp = row.get("TRADETIME") or ""
            side_raw = str(row.get("BUYSELL") or "").upper()
            side = Side.BUY if side_raw == "B" else (Side.SELL if side_raw == "S" else None)
            result.append(
                TradePrint(
                    instrument_uid=meta.instrument_uid or meta.ticker,
                    time_ms=self._msk_time_to_ms(str(stamp)),
                    price=price,
                    quantity=quantity,
                    side=side,
                    trade_id=str(trade_no or ""),
                )
            )
        return result

    async def run(self, duration_seconds: Optional[float] = None, stop_event: Any = None) -> Dict[str, Any]:
        from bot_a.backtest.recorder import MarketDataRecorder
        from bot_a.models.domain import MarketEvent

        recorder = MarketDataRecorder(self._config, self._preset, self._log)
        await recorder.start()
        legs = [(1, self._preset.leg1), (2, self._preset.leg2)]
        deadline = (time.monotonic() + duration_seconds) if duration_seconds else None
        books_available = True
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if stop_event is not None and stop_event.is_set():
                    break
                for leg, meta in legs:
                    if books_available:
                        try:
                            payload = await asyncio.to_thread(_get_json, self._book_url(meta), 30)
                            book = self._parse_book(meta, payload)
                            if book is not None:
                                recorder.write(MarketEvent.of_book(leg, book))
                        except Exception as exc:
                            books_available = False
                            self._log.warning(
                                "Стакан MOEX ISS недоступен (%s) — ресурс orderbook отдаётся только "
                                "подписчикам. Пишем только ленту сделок; для полноценного L2 задайте "
                                "TINVEST_TOKEN (REST-клиент T-Invest).",
                                exc,
                            )
                    try:
                        trades_payload = await asyncio.to_thread(_get_json, self._trades_url(meta), 30)
                        for trade in self._parse_trades(meta, trades_payload):
                            recorder.write(MarketEvent.of_trade(leg, trade))
                    except Exception as exc:  # ISS отвечает 403 при частых запросах
                        self._log.warning("ISS %s: %s", meta.ticker, exc)
                await asyncio.sleep(self._interval / 1000.0)
        finally:
            await recorder.close()
        return recorder.stats.to_dict()
