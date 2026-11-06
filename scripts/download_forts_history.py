"""Загрузка 6 месяцев минутных свечей FORTS через MOEX ISS (кэш `data_cache/`).

Кэш дневной и дополняемый: повторный запуск докачивает только отсутствующие дни.
Пример::

    py -3 scripts/download_forts_history.py
    py -3 scripts/download_forts_history.py --tickers CNYRUBF IMOEXF --from 2026-04-25
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot_a.backtest.iss_source import IssCandleSource, fetch_futures_metadata  # noqa: E402
from bot_a.utils.time_provider import MSK  # noqa: E402

DEFAULT_TICKERS = (
    "CNYRUBF",
    "IMOEXF",
    "GLDRUBF",
    "CRZ6",
    "MMZ6",
    "SRZ6",
    "SVZ6",
    "GDZ6",
    "NGX6",
    "NGZ6",
    "MXZ6",
    "USDRUBF",
    "SiZ6",
)
DEFAULT_FROM = "2026-04-25"
DEFAULT_TO = "2026-11-05"


async def download(tickers: List[str], date_from: str, date_to: str) -> None:
    start = int(datetime.fromisoformat(date_from).replace(tzinfo=MSK).timestamp() * 1000)
    end = int(
        (datetime.fromisoformat(date_to) + timedelta(days=1)).replace(tzinfo=MSK).timestamp()
        * 1000
        - 1
    )
    source = IssCandleSource(ROOT / "data_cache")
    for ticker in tickers:
        began = time.time()
        try:
            meta = fetch_futures_metadata(ticker)
        except Exception as exc:  # noqa: BLE001 — тикер мог истечь
            print(f"{ticker:<9} МЕТАДАННЫЕ НЕДОСТУПНЫ: {type(exc).__name__}: {exc}")
            continue
        candles = await source.get_candles(meta, start, end, "1min")
        days = len({datetime.fromtimestamp(c.time_ms / 1000, tz=MSK).date() for c in candles})
        volume = sum(c.volume for c in candles)
        print(
            f"{ticker:<9} свечей {len(candles):>6}  дней {days:>3}  объём {volume:>12}  "
            f"{time.time() - began:5.1f} c  ГО {meta.initial_margin_buy} RUB",
            flush=True,
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Загрузка истории FORTS с MOEX ISS")
    parser.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    parser.add_argument("--from", dest="date_from", default=DEFAULT_FROM)
    parser.add_argument("--to", dest="date_to", default=DEFAULT_TO)
    args = parser.parse_args(argv)
    asyncio.run(download(args.tickers, args.date_from, args.date_to))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
