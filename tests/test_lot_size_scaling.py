"""Регресс v2.6.1: масштаб лота не должен менять экономику сделки.

1. Две акции с одинаковой рублёвой стоимостью лота (`50 RUB`) и одинаковой комиссией
   обязаны давать одинаковые `Z`, `sigma/Fee`, рублёвую цель тейка и относительный
   отступ тейка по цене.
2. Цель тейка обязана быть в рублях **Leg1** (девиация спреда / бета), а не в
   «спред-рублях»: до v2.6.1 отступ по цене был завышен в β раз.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from bot_a.models.domain import ExecutionTopology, OpenPosition, Side
from bot_a.utils.quotation import dec
from tests.helpers import make_preset, make_spot
from tests.test_engine import _params, _tick


def _engine_for(tmp_path, leg1, leg2, params):
    """Собирает ядро на заданных инструментах (спот, лоты могут отличаться)."""
    from bot_a.economics.fees import TinkoffFeeCalculator
    from bot_a.engine.factory import create_engine, create_guard
    from bot_a.engine.ledger import PortfolioLedger
    from bot_a.execution.simulated_gateway import SimulatedOrderExecutionGateway
    from bot_a.utils.time_provider import SimulatedTimeProvider
    from tests.helpers import make_config, trading_start_ms

    preset = make_preset(leg1, leg2)
    config = make_config(tmp_path, preset, params)
    clock = SimulatedTimeProvider(trading_start_ms())
    fee = TinkoffFeeCalculator(config.fees.tariff, clock=clock)
    gateway = SimulatedOrderExecutionGateway(
        metadata=[leg1, leg2],
        clock=clock,
        fee_calculator=fee,
        latency_ms=0,
        queue_fill_factor=0.35,
        slippage_steps=0,
        order_book_depth=5,
        initial_cash_rub=config.execution.paper_initial_cash_rub,
    )
    guard = create_guard(config, [leg1, leg2])
    ledger = PortfolioLedger(config.execution.paper_initial_cash_rub)
    engine = create_engine(
        [leg1, leg2],
        params,
        ExecutionTopology.SINGLE_LEG_LEAD_LAG,
        gateway,
        fee,
        clock,
        guard,
        ledger,
    )
    return config, clock, fee, gateway, engine, ledger


async def _warm(engine, clock, gateway, price1: Decimal, price2: Decimal, step2: Decimal) -> None:
    """Прогрев с колебанием спреда + шок Leg2, чтобы появились mu, sigma и бета."""
    for index in range(80):
        p2 = price2 if index % 2 == 0 else price2 + step2
        await _tick(engine, clock, gateway, price1, p2)
    for _ in range(3):
        await _tick(engine, clock, gateway, price1, price2 + 40 * step2)


def _position_from_metrics(engine, fee_rub: Decimal, entry_price: Decimal) -> OpenPosition:
    """Позиция с замороженными параметрами входа — как их фиксирует ядро при входе."""
    metrics = engine.metrics
    return OpenPosition(
        trade_id=1,
        opened_ms=engine._clock.now_ms(),  # noqa: SLF001
        side_leg1=Side.SELL,
        quantity=1,
        leg1_entry_price=entry_price,
        leg1_filled_qty=1,
        entry_z=float(metrics.zscore),
        frozen_mu=float(metrics.mean_rub),
        frozen_sigma=float(metrics.variance) ** 0.5,
        frozen_beta=float(metrics.beta),
        entry_spread=float(metrics.spread_rub),
        round_trip_fee_rub=fee_rub,
    )


def _measure(tmp_path, leg1, leg2, params, price1: Decimal, price2: Decimal, step2: Decimal):
    async def scenario():
        config, clock, fee, gateway, engine, ledger = _engine_for(tmp_path, leg1, leg2, params)
        await _warm(engine, clock, gateway, price1, price2, step2)
        metrics = engine.metrics
        fee_rt = engine._current_round_trip_fee(1)  # noqa: SLF001
        position = _position_from_metrics(engine, fee_rt, metrics.best_bid_leg1)
        target = engine._target_tp_rub(position, clock.now_ms())  # noqa: SLF001
        tp_price = engine._tp_price_for(position, target)  # noqa: SLF001
        sigma_leg1 = float(engine._sigma_leg1_rub(metrics))  # noqa: SLF001
        return {
            "z": float(metrics.zscore),
            "sigma": sigma_leg1,
            "fee": float(fee_rt),
            "target": float(target),
            "sigma_to_fee": sigma_leg1 / float(fee_rt) if fee_rt else 0.0,
            "tp_delta": float(abs(tp_price - position.leg1_entry_price)),
            "entry": float(position.leg1_entry_price),
            "beta": position.frozen_beta,
            "spread_deviation": abs(position.entry_spread - position.frozen_mu),
            "lot": leg1.lot_size,
        }

    return asyncio.run(scenario())


def test_lot_size_does_not_change_economics(tmp_path):
    """A (50 RUB, лот 1) и B (5 RUB, лот 10) — одинаковая стоимость лота в рублях."""
    stock_a = make_spot("AAA", lot=1, price_step="0.01", uid="uid-AAA")
    stock_b = make_spot("BBB", lot=10, price_step="0.001", uid="uid-BBB")
    leg2 = make_spot("SBER", lot=1, price_step="0.01", uid="uid-SBER")
    params = _params(beta_mode="slow_ratio")

    a = _measure(tmp_path / "a", stock_a, leg2, params, dec("50.00"), dec("300.00"), dec("0.01"))
    b = _measure(tmp_path / "b", stock_b, leg2, params, dec("5.000"), dec("300.00"), dec("0.01"))

    # расхождение только из-за разной дискретности шага цены (0.01 против 0.001): ~0.1 %
    assert abs(a["z"] - b["z"]) / max(abs(a["z"]), 1e-12) < 5e-3, (a["z"], b["z"])
    assert abs(a["sigma"] - b["sigma"]) / max(abs(a["sigma"]), 1e-12) < 5e-3, (a["sigma"], b["sigma"])
    assert abs(a["fee"] - b["fee"]) / max(a["fee"], 1e-12) < 1e-3, (a["fee"], b["fee"])
    assert abs(a["sigma_to_fee"] - b["sigma_to_fee"]) / max(abs(a["sigma_to_fee"]), 1e-12) < 1e-2

    # рублёвая цель тейка одинакова с точностью до шага цены (у B шаг в 10 раз мельче)
    assert abs(a["target"] - b["target"]) / max(a["target"], 1e-9) < 0.01, (a["target"], b["target"])

    # относительный отступ тейка по цене одинаков: ΔP_A / P_A == ΔP_B / P_B
    rel_a = a["tp_delta"] / a["entry"]
    rel_b = b["tp_delta"] / b["entry"]
    assert abs(rel_a - rel_b) < 5e-3, (rel_a, rel_b)


def test_gross_fee_net_scale_with_lot():
    """ΔP на акцию × лот = рублёвый gross; комиссия берётся с той же суммы."""
    from bot_a.economics.fees import Tariff, TinkoffFeeCalculator

    stock_a = make_spot("AAA", lot=1, price_step="0.01", uid="uid-AAA")
    stock_b = make_spot("BBB", lot=10, price_step="0.001", uid="uid-BBB")
    calc = TinkoffFeeCalculator(Tariff.PREMIUM)

    delta_share = dec("0.05")  # одинаковое относительное движение: 0.1 % от цены
    gross_a = delta_share * stock_a.rub_per_price_unit
    gross_b = (delta_share / 10) * stock_b.rub_per_price_unit
    assert gross_a == gross_b == dec("0.05")

    fee_a = calc.estimated_round_trip_fee(
        stock_a, dec("50.00"), 1, topology=ExecutionTopology.SINGLE_LEG_LEAD_LAG
    )
    fee_b = calc.estimated_round_trip_fee(
        stock_b, dec("5.000"), 1, topology=ExecutionTopology.SINGLE_LEG_LEAD_LAG
    )
    assert fee_a == fee_b, (fee_a, fee_b)
    assert gross_a - fee_a == gross_b - fee_b


def test_tp_target_is_in_leg1_rubles_not_spread_rubles(tmp_path):
    """Цель тейка делится на бету: отступ по цене = девиация спреда / β / лот."""
    leg1 = make_spot("AAA", lot=1, price_step="0.01", uid="uid-AAA")
    leg2 = make_spot("SBER", lot=1, price_step="0.01", uid="uid-SBER")
    params = _params(beta_mode="slow_ratio")

    result = _measure(tmp_path, leg1, leg2, params, dec("50.00"), dec("300.00"), dec("0.01"))

    beta = result["beta"]
    assert beta > 1.5, f"ожидалась бета порядка V2/V1, получено {beta}"
    # Отступ тейка в рублях Leg1 не превышает девиацию спреда, делённую на бету
    # (с точностью до округления вверх до шага цены).
    limit_rub = result["spread_deviation"] / beta
    assert result["tp_delta"] <= limit_rub + 0.011, (result["tp_delta"], limit_rub)
    # и он заметно меньше самой девиации спреда — это и есть проверка деления на бету
    assert result["tp_delta"] < result["spread_deviation"] * 0.9, (
        result["tp_delta"],
        result["spread_deviation"],
    )
