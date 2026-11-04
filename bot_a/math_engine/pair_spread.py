"""Математическое ядро: взвешенный OBI, микропрайс и O(1) EWMA Z-score.

Горячий путь не выполняет ни аллокаций списков, ни пересчёта истории:
на каждый тик пересчитываются только top-k уровней стакана (k = 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from bot_a.models.domain import InstrumentMetadata, OrderBookSnapshot, SpreadMetrics

OBI_DEPTH = 5
BETA_MODES = ("static", "same_underlying", "slow_ratio", "dynamic")
MIN_VARIANCE = 1e-12
_WEIGHT_CACHE: Dict[int, Tuple[float, ...]] = {}
_MSK_OFFSET_MS = 3 * 3600 * 1000


def parse_clock_msk(value: str) -> Optional[int]:
    """``"10:00:15"`` -> миллисекунды от начала суток (MSK). ``None``, если пусто."""
    text = (value or "").strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0
        seconds = int(parts[2]) if len(parts) > 2 else 0
    except (ValueError, IndexError):
        return None
    return ((hours * 60 + minutes) * 60 + seconds) * 1000


def parse_clock_windows(value: str) -> Tuple[Tuple[int, int], ...]:
    """``"13:58-14:05,18:40-19:05"`` -> кортеж интервалов в мс от начала суток."""
    windows: List[Tuple[int, int]] = []
    for chunk in (value or "").split(","):
        chunk = chunk.strip()
        if not chunk or "-" not in chunk:
            continue
        left, right = chunk.split("-", 1)
        start = parse_clock_msk(left)
        end = parse_clock_msk(right)
        if start is None or end is None or end <= start:
            continue
        windows.append((start, end))
    return tuple(windows)


def msk_time_of_day_ms(time_ms: int) -> int:
    """Время суток по MSK в миллисекундах (учитывает только сдвиг UTC+3)."""
    return (int(time_ms) + _MSK_OFFSET_MS) % 86_400_000


def obi_weights(depth: int = OBI_DEPTH) -> Tuple[float, ...]:
    """Веса ``exp(-0.5 * i)`` для уровней стакана, начиная с лучшего."""
    weights = _WEIGHT_CACHE.get(depth)
    if weights is None:
        weights = tuple(math.exp(-0.5 * index) for index in range(depth))
        _WEIGHT_CACHE[depth] = weights
    return weights


@dataclass(frozen=True, slots=True)
class BookStats:
    """Мгновенные метрики одного стакана."""

    best_bid: Decimal
    best_ask: Decimal
    weighted_bid: float
    weighted_ask: float
    obi: float
    microprice: float

    @property
    def mid(self) -> float:
        return (float(self.best_bid) + float(self.best_ask)) / 2.0

    @property
    def spread_bps(self) -> float:
        """Относительный спред стакана в базисных пунктах (широкий = аукцион)."""
        center = self.mid
        if center <= 0:
            return 0.0
        return (float(self.best_ask) - float(self.best_bid)) / center * 10_000.0

    @property
    def spread_steps_value(self) -> float:
        return float(self.best_ask) - float(self.best_bid)


def book_stats(book: OrderBookSnapshot, depth: int = OBI_DEPTH) -> Optional[BookStats]:
    """O(k) пересчёт OBI и микропрайса по top-k уровням."""
    if not book.bids or not book.asks:
        return None
    weights = obi_weights(depth)
    v_bid = 0.0
    v_ask = 0.0
    for index in range(min(depth, len(book.bids))):
        v_bid += book.bids[index].quantity * weights[index]
    for index in range(min(depth, len(book.asks))):
        v_ask += book.asks[index].quantity * weights[index]
    total = v_bid + v_ask
    if total <= 0.0:
        return None
    best_bid = book.bids[0].price
    best_ask = book.asks[0].price
    obi = (v_bid - v_ask) / total
    microprice = (v_bid * float(best_ask) + v_ask * float(best_bid)) / total
    return BookStats(
        best_bid=best_bid,
        best_ask=best_ask,
        weighted_bid=v_bid,
        weighted_ask=v_ask,
        obi=obi,
        microprice=microprice,
    )


class EwmaZScore:
    """Инкрементальный EWMA-дисперсионный Z-score.

    ``alpha = 2 / (N + 1)``; обновление строго O(1), память — O(1).
    """

    __slots__ = ("window", "alpha", "mean", "variance", "count")

    def __init__(self, window: int = 400) -> None:
        if window < 2:
            raise ValueError("EwmaWindow должен быть >= 2")
        self.window = int(window)
        self.alpha = 2.0 / (self.window + 1.0)
        self.mean = 0.0
        self.variance = 0.0
        self.count = 0

    @property
    def is_warm(self) -> bool:
        return self.count >= self.window

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.variance = 0.0

    def update(self, value: float) -> Tuple[float, float, float, bool]:
        """Возвращает ``(mean, variance, zscore, is_warm)`` после обновления."""
        self.count += 1
        if self.count == 1:
            self.mean = value
            self.variance = 0.0
            return self.mean, 0.0, 0.0, False
        previous_mean = self.mean
        alpha = self.alpha
        self.mean = alpha * value + (1.0 - alpha) * previous_mean
        self.variance = (1.0 - alpha) * (self.variance + alpha * (value - previous_mean) ** 2)
        sigma = math.sqrt(max(self.variance, MIN_VARIANCE))
        zscore = (value - self.mean) / sigma
        return self.mean, self.variance, zscore, self.count >= self.window


class PairSpreadCalculator:
    """Синтетический спред пары и Z-score, не зависящий от источника данных.

    Дополнительно ведётся онлайн-оценка динамической беты (отношение
    ковариации к дисперсии изменений рублёвой стоимости Leg1), чтобы ноги
    имели сопоставимую рублёвую чувствительность.
    """

    __slots__ = (
        "_leg1",
        "_leg2",
        "_beta_static",
        "_beta",
        "_beta_min",
        "_beta_max",
        "_beta_min_samples",
        "_beta_mode",
        "_beta_frozen",
        "_beta_observations",
        "_alpha",
        "_slow_alpha",
        "_var1",
        "_cov12",
        "_ret_count",
        "_last_v1",
        "_last_v2",
        "_mean_v1",
        "_mean_v2",
        "_macro_alpha",
        "_macro_v1",
        "_macro_v2",
        "_macro_count",
        "_macro_min_samples",
        "_beta_macro",
        "_stat_interval_ms",
        "_last_stat_ms",
        "_last_beta_ms",
        "_obi_depth",
        "_ewma",
        "_books",
        "_stats",
        "_last",
        "_pending",
        "_freeze_until_ms",
        "_freeze_windows",
        "_max_spread_bps",
        "_freeze_reason",
        "_frozen_ticks",
        "_gate_bypass",
    )

    def __init__(
        self,
        leg1: InstrumentMetadata,
        leg2: InstrumentMetadata,
        beta: float = 1.0,
        ewma_window: int = 400,
        obi_depth: int = OBI_DEPTH,
        beta_mode: str = "same_underlying",
        use_dynamic_beta: Optional[bool] = None,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        beta_min_samples: int = 30,
        macro_beta_window: int = 14400,
        macro_beta_min_samples: int = 1800,
        stat_sample_interval_seconds: float = 0.0,
        stat_freeze_until_msk: str = "",
        stat_freeze_windows: str = "",
        max_valid_spread_bps: float = 0.0,
    ) -> None:
        self._leg1 = leg1
        self._leg2 = leg2
        self._beta_static = float(beta)
        self._beta = float(beta)
        self._beta_min = float(beta_min)
        self._beta_max = float(beta_max)
        self._beta_min_samples = max(int(beta_min_samples), 2)
        # Обратная совместимость: use_dynamic_beta=False включает статическую бету.
        if use_dynamic_beta is False and beta_mode == "same_underlying":
            beta_mode = "static"
        elif use_dynamic_beta is True and beta_mode == "same_underlying":
            beta_mode = "dynamic"
        self._beta_mode = beta_mode if beta_mode in BETA_MODES else "same_underlying"
        self._beta_frozen = False
        self._beta_observations = 0
        self._alpha = 2.0 / (int(ewma_window) + 1.0)
        self._slow_alpha = self._alpha
        self._var1 = 0.0
        self._cov12 = 0.0
        self._ret_count = 0
        self._last_v1: Optional[float] = None
        self._last_v2: Optional[float] = None
        self._mean_v1 = 0.0
        self._mean_v2 = 0.0
        self._macro_alpha = 2.0 / (max(int(macro_beta_window), 2) + 1.0)
        self._macro_v1 = 0.0
        self._macro_v2 = 0.0
        self._macro_count = 0
        self._macro_min_samples = max(int(macro_beta_min_samples), 2)
        self._beta_macro = 0.0
        # v2.4.1: квантование статистических окон по времени события.
        # 0 — обновлять на каждом снимке стакана (бэктест: поведение v2.4 без изменений).
        interval = max(float(stat_sample_interval_seconds), 0.0)
        self._stat_interval_ms = int(round(interval * 1000.0))
        self._last_stat_ms: Optional[int] = None
        self._last_beta_ms: Optional[int] = None
        self._obi_depth = int(obi_depth)
        self._ewma = EwmaZScore(ewma_window)
        self._books: List[Optional[OrderBookSnapshot]] = [None, None]
        self._stats: List[Optional[BookStats]] = [None, None]
        self._last: Optional[SpreadMetrics] = None
        self._pending = [False, False]
        # v2.7: окна, в которых статистика (mu/sigma/beta/regime) не двигается.
        parsed = parse_clock_msk(stat_freeze_until_msk)
        self._freeze_until_ms = parsed if parsed is not None else 0
        self._freeze_windows = parse_clock_windows(stat_freeze_windows)
        self._max_spread_bps = max(float(max_valid_spread_bps), 0.0)
        self._freeze_reason: Optional[str] = None
        self._frozen_ticks = 0
        self._gate_bypass = False

    def set_stat_gate_bypass(self, value: bool) -> None:
        """v2.8: отключить гейт пре-маркета (только для прогрева историей)."""
        self._gate_bypass = bool(value)

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def static_beta(self) -> float:
        return self._beta_static

    @property
    def beta_mode(self) -> str:
        return self._beta_mode

    @property
    def beta_frozen(self) -> bool:
        return self._beta_frozen

    @property
    def beta_macro(self) -> float:
        """Сверхмедленная (макро) бета — базовая линия структурного дрейфа."""
        return self._beta_macro

    @property
    def beta_observations(self) -> int:
        """Сколько снимков участвовало в оценке активной беты."""
        return self._beta_observations

    @property
    def macro_samples(self) -> int:
        """Сколько снимков набрала макро-бета (нужно ``macroBetaMinSamples``)."""
        return self._macro_count

    @property
    def stat_sample_interval_seconds(self) -> float:
        return self._stat_interval_ms / 1000.0

    @property
    def freeze_reason(self) -> Optional[str]:
        """Причина, по которой последний тик не двигал статистику (``None`` — не было)."""
        return self._freeze_reason

    @property
    def frozen_ticks(self) -> int:
        """Сколько тиков отброшено из статистики по гейту v2.7."""
        return self._frozen_ticks

    def _stat_gate(
        self,
        time_ms: int,
        stats1: Optional[BookStats],
        stats2: Optional[BookStats],
    ) -> Optional[str]:
        """Почему тик нельзя пускать в статистику (v2.7).

        Три причины, каждая из которых на живом рынке портила базу:

        * пре-маркет и аукцион открытия — до ``stat_freeze_until_msk`` (10:00:15 MSK);
          именно оттуда брался ``|Z| = 12.98`` в 09:20 у ``AFKS``;
        * клиринговые аукционы (13:58–14:05, 18:40–19:05) — те же широкие котировки;
        * аномально широкий спред стакана (``max_valid_spread_bps``) — признак аукциона
          или дырки в книге, по которой нельзя считать спред пары.
        """
        if stats1 is None or stats2 is None:
            return "нет стакана"
        if self._gate_bypass:
            return None
        if self._freeze_until_ms or self._freeze_windows:
            minute_of_day = msk_time_of_day_ms(time_ms)
            if self._freeze_until_ms and minute_of_day < self._freeze_until_ms:
                return "пре-маркет (до 10:00:15 MSK)"
            for start, end in self._freeze_windows:
                if start <= minute_of_day < end:
                    return "клиринговый аукцион"
        if self._max_spread_bps > 0:
            widest = max(stats1.spread_bps, stats2.spread_bps)
            if widest > self._max_spread_bps:
                return f"широкий стакан ({widest:.0f} bps)"
        return None

    def _stat_due(self, time_ms: int) -> bool:
        """Пора ли сдвинуть статистические окна (v2.4.1).

        При ``stat_sample_interval_seconds = 0`` окна двигаются на каждом снимке
        стакана — это исторический путь бэктестера, сохранённый бит-в-бит.
        """
        if self._stat_interval_ms <= 0:
            return True
        last = self._last_stat_ms
        if last is None or time_ms - last >= self._stat_interval_ms:
            return True
        return False

    def _beta_due(self, time_ms: int) -> bool:
        """Пора ли двигать бету и макро-бету (v2.6.7).

        Считается по собственному таймеру: согласованный снимок обеих ног может
        прийти в «несэмпловый» тик, когда ноги фазово синхронизированы и несут
        одну метку времени. Каденция ``stat_sample_interval_seconds`` сохраняется.
        """
        if self._stat_interval_ms <= 0:
            return True
        last = self._last_beta_ms
        if last is None or time_ms - last >= self._stat_interval_ms:
            self._last_beta_ms = int(time_ms)
            return True
        return False

    @property
    def beta_drift(self) -> Optional[float]:
        """Относительный дрейф активной беты к макро-базе (``None`` до прогрева).

        ``> 0`` — Leg1 структурно слабеет относительно Leg2 (Leg1 дешевеет),
        ``< 0`` — Leg1 структурно растёт относительно Leg2.
        """
        if self._macro_count < self._macro_min_samples or self._beta_macro <= 1e-12:
            return None
        return (self._beta - self._beta_macro) / self._beta_macro

    def freeze_beta(self) -> None:
        """Фиксирует бету на время сделки (вход, котирование, удержание)."""
        self._beta_frozen = True

    def unfreeze_beta(self) -> None:
        self._beta_frozen = False

    @property
    def ewma(self) -> EwmaZScore:
        return self._ewma

    @property
    def last_metrics(self) -> Optional[SpreadMetrics]:
        return self._last

    @property
    def leg1(self) -> InstrumentMetadata:
        return self._leg1

    @property
    def leg2(self) -> InstrumentMetadata:
        return self._leg2

    def reset(self) -> None:
        self._ewma.reset()
        self._books = [None, None]
        self._stats = [None, None]
        self._last = None
        self._beta = self._beta_static
        self._var1 = 0.0
        self._cov12 = 0.0
        self._ret_count = 0
        self._last_v1 = None
        self._last_v2 = None
        self._pending = [False, False]
        self._beta_frozen = False
        self._beta_observations = 0
        self._mean_v1 = 0.0
        self._mean_v2 = 0.0
        self._macro_v1 = 0.0
        self._macro_v2 = 0.0
        self._macro_count = 0
        self._beta_macro = 0.0
        self._last_stat_ms = None
        self._last_beta_ms = None

    def _update_beta(self, value1: float, value2: float) -> None:
        """Обновляет бету согласно выбранному режиму (см. ``BETA_MODES``).

        * ``static`` / ``same_underlying`` — фиксированная бета (одна и та же нога);
        * ``slow_ratio`` — медленные средние EWMA рублёвых стоимостей ног,
          нормирующие обе ноги к равному риску;
        * ``dynamic`` — устаревшая оценка cov(dV2,dV1)/var(dV1) по тикам (страдала
          эффектом Эппса и была источником скачков в v2; оставлена для сравнения).
        """
        self._beta_observations += 1
        if self._beta_mode in ("static", "same_underlying"):
            self._beta = self._beta_static
            return
        if self._beta_frozen:
            return
        if self._beta_mode == "slow_ratio":
            if self._beta_observations == 1:
                self._mean_v1 = value1
                self._mean_v2 = value2
                return
            alpha = self._slow_alpha
            self._mean_v1 = alpha * value1 + (1.0 - alpha) * self._mean_v1
            self._mean_v2 = alpha * value2 + (1.0 - alpha) * self._mean_v2
            if self._beta_observations >= self._beta_min_samples and self._mean_v1 > 1e-9:
                self._set_beta(self._mean_v2 / self._mean_v1)
            return
        # dynamic (legacy)
        if self._last_v1 is None or self._last_v2 is None:
            self._last_v1, self._last_v2 = value1, value2
            return
        delta1 = value1 - self._last_v1
        delta2 = value2 - self._last_v2
        alpha = self._alpha
        self._var1 = (1.0 - alpha) * (self._var1 + alpha * delta1 * delta1)
        self._cov12 = (1.0 - alpha) * (self._cov12 + alpha * delta1 * delta2)
        self._ret_count += 1
        if self._ret_count >= self._beta_min_samples and self._var1 > 1e-12:
            self._set_beta(self._cov12 / self._var1)
        self._last_v1, self._last_v2 = value1, value2

    def _update_macro_beta(self, value1: float, value2: float) -> None:
        """Сверхмедленная бета (макро-база) для оценки структурного дрейфа.

        Считается всегда, независимо от заморозки активной беты: это индикатор
        режима, а не параметр нормировки спреда.
        """
        self._macro_count += 1
        if self._macro_count == 1:
            self._macro_v1 = value1
            self._macro_v2 = value2
        else:
            alpha = self._macro_alpha
            self._macro_v1 = alpha * value1 + (1.0 - alpha) * self._macro_v1
            self._macro_v2 = alpha * value2 + (1.0 - alpha) * self._macro_v2
        if self._macro_v1 > 1e-9:
            self._beta_macro = self._macro_v2 / self._macro_v1

    def _set_beta(self, beta: float) -> None:
        if beta < self._beta_min:
            beta = self._beta_min
        elif beta > self._beta_max:
            beta = self._beta_max
        self._beta = beta

    def on_book(self, leg_index: int, book: OrderBookSnapshot) -> Optional[SpreadMetrics]:
        """Обновляет ногу ``leg_index`` (0 — Leg 1, 1 — Leg 2).

        Возвращает метрики только когда собраны оба стакана.
        """
        if leg_index not in (0, 1):
            raise ValueError("leg_index должен быть 0 или 1")
        stats = book_stats(book, self._obi_depth)
        if stats is None:
            return None
        self._books[leg_index] = book
        self._stats[leg_index] = stats
        # Флаги ставятся сразу после обновления ноги: бета считается только по
        # согласованному снимку, когда обе ноги обновились с прошлого расчёта.
        self._pending[leg_index] = True
        stats1 = self._stats[0]
        stats2 = self._stats[1]
        if stats1 is None or stats2 is None:
            return None

        value1 = float(self._leg1.value_in_rub(Decimal(repr(stats1.microprice))))
        value2 = float(self._leg2.value_in_rub(Decimal(repr(stats2.microprice))))
        gate = self._stat_gate(book.time_ms, stats1, stats2)
        self._freeze_reason = gate
        if gate is not None:
            self._frozen_ticks += 1
        sampled = gate is None and self._stat_due(book.time_ms)
        if self._pending[0] and self._pending[1]:
            self._pending = [False, False]
            if sampled:
                self._update_beta(value1, value2)
                self._update_macro_beta(value1, value2)
        spread_rub = value2 - self._beta * value1
        if sampled:
            mean, variance, zscore, is_warm = self._ewma.update(spread_rub)
            self._last_stat_ms = book.time_ms
        else:
            # Мгновенный Z считается по текущим (не обновлённым) mu/sigma:
            # заявки и стопы видят реальное состояние спреда на каждом тике L2.
            mean = self._ewma.mean
            variance = self._ewma.variance
            is_warm = self._ewma.is_warm
            if self._ewma.count <= 0:
                zscore = 0.0
            else:
                sigma = math.sqrt(max(variance, MIN_VARIANCE))
                zscore = (spread_rub - mean) / sigma
        if gate is not None and self._ewma.count > 0:
            # v2.7: на аукционе спред считается «неопределённым» — иначе в лог,
            # трейл и дашборд попадают |Z| вроде 12.98 из пре-маркета.
            zscore = 0.0

        metrics = SpreadMetrics(
            time_ms=book.time_ms,
            spread_rub=spread_rub,
            mean_rub=mean,
            variance=variance,
            zscore=zscore,
            obi_leg1=stats1.obi,
            obi_leg2=stats2.obi,
            micro_leg1=Decimal(repr(stats1.microprice)),
            micro_leg2=Decimal(repr(stats2.microprice)),
            best_bid_leg1=stats1.best_bid,
            best_ask_leg1=stats1.best_ask,
            best_bid_leg2=stats2.best_bid,
            best_ask_leg2=stats2.best_ask,
            is_warm=is_warm,
            tick_count=self._ewma.count,
            beta=self._beta,
            beta_macro=self._beta_macro,
            beta_drift=self.beta_drift,
            stat_sampled=sampled,
            stat_frozen=gate is not None,
            freeze_reason=gate,
        )
        self._last = metrics
        return metrics
