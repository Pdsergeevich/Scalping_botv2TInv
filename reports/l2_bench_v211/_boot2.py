import sys, asyncio
sys.path.insert(0, ".")
from bot_a.config import AppConfig
from bot_a.backtest.historical_candles import iter_candle_events
from bot_a.backtest.iss_source import IssCandleSource
from bot_a.math_engine.pair_spread import PairSpreadCalculator

cfg = AppConfig.load("appsettings.json")
s = cfg.strategy

async def main():
    for pair in ("VTBR_SBER", "MTLR_SBER", "IRAO_SBER"):
        preset = cfg.preset(pair)
        src = IssCandleSource(cfg.path(cfg.storage.data_cache_dir))
        import time
        end = int(time.time() * 1000)
        start = end - 8 * 24 * 3600 * 1000
        c1, c2 = await asyncio.gather(
            src.get_candles(preset.leg1, start, end, "1min"),
            src.get_candles(preset.leg2, start, end, "1min"),
        )
        calc = PairSpreadCalculator(
            leg1=preset.leg1, leg2=preset.leg2, beta=float(preset.beta),
            ewma_window=s.ewma_window, obi_depth=s.obi_depth, beta_mode=preset.beta_mode,
            beta_min=s.beta_min, beta_max=s.beta_max, beta_min_samples=s.beta_min_samples,
            macro_beta_window=s.macro_beta_window, macro_beta_min_samples=s.macro_beta_min_samples,
            stat_sample_interval_seconds=s.stat_sample_interval_seconds,
        )
        for ev in iter_candle_events(preset.leg1, preset.leg2, c1, c2, depth=10, phase_lock=True, synth_mode="ticks"):
            if ev.is_book and ev.book is not None:
                calc.on_book(ev.leg - 1, ev.book)
        print(f"{pair:<11} свечей {len(c1)}/{len(c2)} beta={calc.beta:8.4f} obs={calc.beta_observations:6d} "
              f"mv1={calc._mean_v1:9.3f} mv2={calc._mean_v2:9.3f} macro={calc.beta_macro:8.4f} ticks={calc.ewma.count}")
asyncio.run(main())

