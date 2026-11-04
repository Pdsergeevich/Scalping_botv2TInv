"""Тесты paper-реплея на свечах ISS: режим синтеза обязан доезжать до ядра.

Баг, который эти тесты закрывают (v2.6.4): `PaperReplayService` жёстко вызывал
`iter_candle_events(..., synth_mode="ticks")`, поэтому `--synth-mode close`
для `--mode paper --source iss` молча игнорировался — прогон «close» давал
байт-в-байт те же цифры, что и «ticks», и сравнение с живым L2 было недостоверным.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from bot_a.backtest.candles import Candle, InMemoryCandleSource
from bot_a.services import paper_replay
from bot_a.services.paper_replay import PaperReplayService
from tests.helpers import make_config, make_meta, make_preset
from tests.test_engine import _params

START_MS = 1_700_000_000_000
END_MS = START_MS + 20 * 60_000


def _candles(count: int = 20) -> list[Candle]:
    return [
        Candle(
            time_ms=START_MS + index * 60_000,
            open=Decimal("100.00"),
            high=Decimal("100.50"),
            low=Decimal("99.50"),
            close=Decimal("100.00") + Decimal(index) / 100,
            volume=10,
        )
        for index in range(count)
    ]


def _service(tmp_path, synth_mode: str) -> PaperReplayService:
    leg1 = make_meta("AAA", price_step="0.01", step_amount="0.01", uid="uid-AAA")
    leg2 = make_meta("BBB", price_step="0.01", step_amount="0.01", uid="uid-BBB")
    preset = make_preset(leg1, leg2)
    config = make_config(tmp_path, preset, _params())
    source = InMemoryCandleSource({"AAA": _candles(), "BBB": _candles()})
    return PaperReplayService(config, preset, source=source, synth_mode=synth_mode)


@pytest.mark.parametrize("mode", ["ticks", "close"])
def test_synth_mode_reaches_event_generator(tmp_path, monkeypatch, mode):
    """Режим из CLI попадает в `iter_candle_events`, а не подменяется на ticks."""
    seen: list[str] = []

    def fake_iter(*args, **kwargs):  # noqa: ANN002, ANN003
        seen.append(kwargs.get("synth_mode"))
        return iter(())

    monkeypatch.setattr(paper_replay, "iter_candle_events", fake_iter)
    service = _service(tmp_path, mode)

    asyncio.run(service.run(START_MS, END_MS, warmup_days=0))

    assert seen == [mode]


def test_unknown_synth_mode_falls_back_to_ticks(tmp_path):
    assert _service(tmp_path, "tickz")._synth_mode == "ticks"  # noqa: SLF001
    assert _service(tmp_path, "close")._synth_mode == "close"  # noqa: SLF001


def test_close_mode_generates_fewer_events_than_ticks(tmp_path):
    """`close` — один тик на бар, `ticks` — 12 подтиков: мост обязан давать больше событий."""
    ticks_result, _, _ = asyncio.run(
        _service(tmp_path / "ticks", "ticks").run(START_MS, END_MS, warmup_days=0)
    )
    close_result, _, _ = asyncio.run(
        _service(tmp_path / "close", "close").run(START_MS, END_MS, warmup_days=0)
    )

    assert ticks_result.events_processed > close_result.events_processed
    assert close_result.events_processed > 0


# ------------------------------------------- v2.6.5: приведение ВСЕХ окон, а не только EWMA
def test_close_mode_aligns_every_time_window(tmp_path):
    """В `close` окна меряются барами: EWMA, макро-бета, порог сэмплов и окно режима."""
    from bot_a.backtest.historical_candles import align_close_mode_params
    from bot_a.engine.fsm import StrategyParams

    params = StrategyParams(
        ewma_window=2400,
        macro_beta_window=14400,
        macro_beta_min_samples=1800,
        regime_window=12,
    )
    ticks_params = align_close_mode_params(params, "ticks", 5.0)
    assert ticks_params is params, "в режиме ticks окна не трогаем"

    close_params = align_close_mode_params(params, "close", 5.0)
    assert close_params.ewma_window == 200
    assert close_params.macro_beta_window == 1200
    assert close_params.macro_beta_min_samples == 150
    assert close_params.regime_window == 2


def test_close_mode_never_shrinks_windows_below_floor():
    """Полы защищают от вырожденных окон на очень маленьких значениях."""
    from bot_a.backtest.historical_candles import align_close_mode_params
    from bot_a.engine.fsm import StrategyParams

    tiny = StrategyParams(
        ewma_window=1, macro_beta_window=1, macro_beta_min_samples=1, regime_window=1
    )
    aligned = align_close_mode_params(tiny, "close", 5.0)
    assert aligned.ewma_window == 10
    assert aligned.macro_beta_window == 100
    assert aligned.macro_beta_min_samples == 20
    assert aligned.regime_window == 2


def test_paper_replay_applies_aligned_windows(tmp_path):
    """Реплей в `close` обязан получить выровненные окна, а не только EWMA."""
    service = _service(tmp_path, "close")
    assert service._params.ewma_window == 10  # 40 / 12 = 3.3, но пол 10 баров
    assert service._params.macro_beta_window == 1200
    assert service._params.macro_beta_min_samples == 150
    assert service._params.regime_window == 2
