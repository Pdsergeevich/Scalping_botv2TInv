"""Тесты резолва метаданных живого фронт-контракта (watch/paper/live) и бар-минут."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import pytest

from bot_a.cli import _resolve_live_metas
from bot_a.models.domain import InstrumentMetadata, MarketKind


class _Logger:
    def __init__(self) -> None:
        self.warnings: List[str] = []

    def warning(self, message: str, *args: Any) -> None:
        self.warnings.append(message % args if args else message)


def _meta(ticker: str, uid: str | None = None) -> InstrumentMetadata:
    return InstrumentMetadata(
        instrument_uid=uid or ticker,
        ticker=ticker,
        class_code="SPBFUT",
        market=MarketKind.FUTURES,
        min_price_increment=Decimal("1"),
        min_price_increment_amount=Decimal("1"),
        lot_size=1,
        initial_margin_buy=Decimal("1000"),
        initial_margin_sell=Decimal("1000"),
        name=ticker,
    )


def test_resolve_live_metas_swaps_continuous_for_front(tmp_path: Path) -> None:
    calls: List[str] = []

    def fetch(ticker: str) -> InstrumentMetadata:
        calls.append(ticker)
        return _meta(ticker, uid=f"uid-{ticker}")

    metas = {"Eu_CONT": _meta("Eu_CONT", uid="uid-Eu_CONT")}
    resolved = asyncio.run(
        _resolve_live_metas(metas, {"Eu_CONT": "EuZ6"}, fetch, _Logger(), cache_dir=tmp_path)
    )
    assert calls == ["EuZ6"], "метаданные должны запрашиваться по живому контракту"
    assert list(resolved) == ["EuZ6"]
    assert resolved["EuZ6"].instrument_uid == "uid-EuZ6"


def test_resolve_live_metas_keeps_plain_tickers(tmp_path: Path) -> None:
    def fetch(ticker: str) -> InstrumentMetadata:  # не должен вызываться
        raise AssertionError("для перпетуала сеть не нужна")

    metas = {"SLVRUBF": _meta("SLVRUBF")}
    resolved = asyncio.run(
        _resolve_live_metas(metas, {"SLVRUBF": "SLVRUBF"}, fetch, _Logger(), cache_dir=tmp_path)
    )
    assert resolved["SLVRUBF"].ticker == "SLVRUBF"


def test_resolve_live_metas_falls_back_to_proxy(tmp_path: Path) -> None:
    def fetch(ticker: str) -> InstrumentMetadata:
        raise RuntimeError("ISS недоступен")

    logger = _Logger()
    metas = {"GL_CONT": _meta("GL_CONT")}
    resolved = asyncio.run(
        _resolve_live_metas(metas, {"GL_CONT": "GLZ6"}, fetch, logger, cache_dir=tmp_path)
    )
    assert resolved["GLZ6"].ticker == "GL_CONT"  # прокси от пресета
    assert logger.warnings, "отсутствие метаданных живого контракта должно логироваться"
