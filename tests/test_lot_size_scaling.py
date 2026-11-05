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


async def _warm(
    engine,
    clock,
    gateway,
    price1: Decimal,
    price2: Decimal,
    step2: Decimal,
    shock_steps: int = 40,
) -> None:
    """Прогрев с колебанием спреда + шок Leg2, чтобы появились mu, sigma и бета."""
    for index in range(80):
        p2 = price2 if index % 2 == 0 else price2 + step2
        await _tick(engine, clock, gateway, price1, p2)
    for _ in range(3):
        await _tick(engine, clock, gateway, price1, price2 + shock_steps * step2)


def _position_from_metrics(
    engine, fee_rub: Decimal, entry_price: Decimal, lots: int = 1
) -> OpenPosition:
    """Позиция с замороженными параметрами входа — как их фиксирует ядро при входе."""
    metrics = engine.metrics
    return OpenPosition(
        trade_id=1,
        opened_ms=engine._clock.now_ms(),  # noqa: SLF001
        side_leg1=Side.SELL,
        quantity=lots,
        leg1_entry_price=entry_price,
        leg1_filled_qty=lots,
        entry_z=float(metrics.zscore),
        frozen_mu=float(metrics.mean_rub),
        frozen_sigma=float(metrics.variance) ** 0.5,
        frozen_beta=float(metrics.beta),
        entry_spread=float(metrics.spread_rub),
        round_trip_fee_rub=fee_rub,
    )


def _measure(
    tmp_path, leg1, leg2, params, price1: Decimal, price2: Decimal, step2: Decimal,
    shock_steps: int = 40,
):
    async def scenario():
        config, clock, fee, gateway, engine, ledger = _engine_for(tmp_path, leg1, leg2, params)
        await _warm(engine, clock, gateway, price1, price2, step2, shock_steps)
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

    # Шок 200 шагов: девиационная ветка цели должна доминировать над «полом комиссии»
    # (после v2.11 EWMA пересевается, и mu корректна — на малом шоке цель упирается
    # в 2.5 × комиссию, что делает проверку деления на бету бессмысленной).
    result = _measure(
        tmp_path, leg1, leg2, params, dec("50.00"), dec("300.00"), dec("0.01"), shock_steps=200
    )

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


# ---------------------------------------------------------------- v2.6.4: инвариантность по Q
async def _prints(engine, clock, gateway, mid1: Decimal, mid2: Decimal, quantity: int = 1000) -> None:
    """Тик с крупным принтом, чтобы позиция любого размера наливалась целиком за один тик."""
    from tests.helpers import make_book, make_trade

    clock.advance(1000)
    await engine.on_clock()
    book1 = make_book(mid1, clock.now_ms(), uid="uid-AAA")
    book2 = make_book(mid2, clock.now_ms(), uid="uid-SBER")
    await engine.on_order_book(1, book1)
    await engine.on_order_book(2, book2)
    await engine.on_trade(1, make_trade(book1.bids[0].price, clock.now_ms(), quantity=quantity, uid="uid-AAA"))
    await engine.on_trade(2, make_trade(book2.bids[0].price, clock.now_ms(), quantity=quantity, uid="uid-SBER"))


async def _engine_with_eager_fills(tmp_path, lots: int):
    """Ядро с мгновенными полными исполнениями: чтобы Q не влиял на микроструктуру налива."""
    from bot_a.economics.fees import TinkoffFeeCalculator
    from bot_a.engine.factory import create_engine, create_guard
    from bot_a.engine.ledger import PortfolioLedger
    from bot_a.execution.simulated_gateway import SimulatedOrderExecutionGateway
    from bot_a.utils.time_provider import SimulatedTimeProvider
    from tests.helpers import make_config, trading_start_ms

    leg1 = make_spot("AAA", lot=1, price_step="0.01", uid="uid-AAA")
    leg2 = make_spot("SBER", lot=1, price_step="0.01", uid="uid-SBER")
    params = _params(quantity_lots=lots, beta_mode="slow_ratio")
    config = make_config(tmp_path, make_preset(leg1, leg2), params)
    clock = SimulatedTimeProvider(trading_start_ms())
    fee = TinkoffFeeCalculator(config.fees.tariff, clock=clock)
    gateway = SimulatedOrderExecutionGateway(
        metadata=[leg1, leg2],
        clock=clock,
        fee_calculator=fee,
        latency_ms=0,
        queue_fill_factor=1.0,
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
    engine._guard._max_lots = lots  # noqa: SLF001 — иначе предохранитель урежет Q до 2
    await gateway.start()
    return leg1, params, clock, fee, gateway, engine, ledger


async def _signal_path(tmp_path, lots: int) -> dict:
    """Один и тот же путь цен при разном Q: счётчики фильтров обязаны совпасть."""
    leg1, params, clock, fee, gateway, engine, _ledger = await _engine_with_eager_fills(tmp_path, lots)
    for index in range(80):
        p2 = dec("101.00") if index % 2 == 0 else dec("101.02")
        await _prints(engine, clock, gateway, dec("100.00"), p2)
    metrics = engine.metrics
    fee_rt = engine._current_round_trip_fee(lots)  # noqa: SLF001
    sigma_leg1 = engine._sigma_leg1_rub(metrics) * Decimal(lots)  # noqa: SLF001
    from bot_a.economics.fees import TinkoffFeeCalculator

    expected_one_lot = Decimal(
        TinkoffFeeCalculator.expected_gross_reversion_rub(
            metrics.spread_rub, metrics.mean_rub, float(metrics.zscore), params.z_exit
        )
    ) / Decimal(str(metrics.beta or 1.0))
    for _ in range(3):
        await _prints(engine, clock, gateway, dec("100.00"), dec("104.00"))
    counters = dict(engine.counters)
    position = engine.position
    return {
        "counters": counters,
        "sigma_to_fee": float(sigma_leg1 / fee_rt) if fee_rt else 0.0,
        "fee": float(fee_rt),
        "expected_one_lot": float(expected_one_lot),
        "entry_ms": position.opened_ms if position is not None else None,
        "entry_price": position.leg1_entry_price if position is not None else None,
        "filled": position.leg1_filled_qty if position is not None else 0,
    }


def test_quantity_lots_invariance(tmp_path):
    """Q=1 и Q=60 обязаны давать одинаковую логику фильтров и одинаковую экономику ×60.

    Регресс 27.10.2026: `expected_gross` и цель тейка считались на 1 лот, а комиссия,
    слиппедж и стоимость шага — на всю позицию. Из-за этого пары с `quantityLots > 1`
    (VTBR 55, MTLR 60) не могли войти: фильтр `edge` отсекал 100 % сигналов.
    """
    one = asyncio.run(_signal_path(tmp_path / "q1", 1))
    many = asyncio.run(_signal_path(tmp_path / "q60", 60))

    # 1) фильтры: ровно те же решения при том же рынке
    for key in ("signals", "edgeSkips", "volFilterSkips", "entries", "hookSkips", "regimeSkips"):
        assert one["counters"].get(key, 0) == many["counters"].get(key, 0), (
            key,
            one["counters"].get(key),
            many["counters"].get(key),
        )
    assert one["counters"]["entries"] >= 1, "сценарий обязан доходить до входа"

    # 2) sigma/Fee не зависит от размера позиции
    assert abs(one["sigma_to_fee"] - many["sigma_to_fee"]) < 1e-9

    # 3) вход — на том же тике и по той же цене, объём — ровно в 60 раз больше
    assert one["entry_ms"] == many["entry_ms"]
    assert one["entry_price"] == many["entry_price"]
    assert one["filled"] == 1 and many["filled"] == 60

    # 4) демонстрация самого бага: 1-лотовый возврат не покрывает 60-лотовую комиссию
    fee_many = many["fee"]
    assert many["expected_one_lot"] < fee_many, (
        "ожидаемый возврат 1 лота обязан быть меньше 60-лотовой комиссии — "
        "именно поэтому старый код блокировал все входы"
    )
    assert abs(many["fee"] - 60 * one["fee"]) < 1e-9


def test_quantity_lots_tp_and_money_scale(tmp_path):
    """Цель тейка ×Q, а цена тейка — та же: множитель Q сокращается при переводе в цену."""
    results = {}
    for lots in (1, 60):
        leg1, params, clock, fee, gateway, engine, _ledger = asyncio.run(
            _engine_with_eager_fills(tmp_path / f"tp{lots}", lots)
        )

        async def scenario() -> dict:
            for index in range(80):
                p2 = dec("101.00") if index % 2 == 0 else dec("101.02")
                await _prints(engine, clock, gateway, dec("100.00"), p2)
            for _ in range(3):
                await _prints(engine, clock, gateway, dec("100.00"), dec("104.00"))
            metrics = engine.metrics
            fee_rt = engine._current_round_trip_fee(lots)  # noqa: SLF001
            position = _position_from_metrics(engine, fee_rt, metrics.micro_leg1, lots=lots)
            target = engine._target_tp_rub(position, clock.now_ms())  # noqa: SLF001
            tp_price = engine._tp_price_for(position, target)  # noqa: SLF001
            return {
                "target": target,
                "tp_price": tp_price,
                "fee": fee_rt,
                "delta": abs(tp_price - position.leg1_entry_price),
                "rub_per_unit": leg1.rub_per_price_unit,
            }

        results[lots] = asyncio.run(scenario())

    small, big = results[1], results[60]
    # цена тейка одинаковая: Q сокращается в _tp_price_for
    assert small["tp_price"] == big["tp_price"], (small["tp_price"], big["tp_price"])
    assert small["delta"] == big["delta"]
    # рублёвая цель и комиссия масштабируются ровно в 60 раз
    assert abs(float(big["target"]) - 60 * float(small["target"])) < 1e-6
    assert abs(float(big["fee"]) - 60 * float(small["fee"])) < 1e-9

    # gross/fees/net на одинаковом ходе цены масштабируются ровно в 60 раз
    for lots, row in results.items():
        gross = row["delta"] * row["rub_per_unit"] * Decimal(lots)
        rows = (gross, row["fee"], gross - row["fee"])
        results[lots]["money"] = rows
    for small_value, big_value in zip(results[1]["money"], results[60]["money"]):
        assert abs(float(big_value) - 60 * float(small_value)) < 1e-6, (small_value, big_value)
