import sys
sys.path.insert(0, ".")
from pathlib import Path
from bot_a.backtest.tick_replay import JsonlRecordingSource
from bot_a.config import AppConfig
from bot_a.math_engine.pair_spread import PairSpreadCalculator
from bot_a.utils.time_provider import ms_to_msk
import asyncio

cfg = AppConfig.load("appsettings.json").with_preset_overrides()
preset = cfg.preset("VTBR_SBER")
s = cfg.strategy
print("beta_min_samples", s.beta_min_samples, "interval", s.stat_sample_interval_seconds, "mode", s.beta_mode)
calc = PairSpreadCalculator(
    leg1=preset.leg1, leg2=preset.leg2, beta=s.beta,
    ewma_window=s.ewma_window, obi_depth=s.obi_depth, beta_mode=s.beta_mode,
    beta_min=s.beta_min, beta_max=s.beta_max, beta_min_samples=s.beta_min_samples,
    macro_beta_window=s.macro_beta_window, macro_beta_min_samples=s.macro_beta_min_samples,
    stat_sample_interval_seconds=s.stat_sample_interval_seconds,
    stat_freeze_until_msk=s.stat_freeze_until_msk, stat_freeze_windows=s.stat_freeze_windows,
    max_valid_spread_bps=s.max_valid_spread_bps,
)
path = Path("recordings/VTBR_SBER_2026-11-05.jsonl.gz")
src = JsonlRecordingSource([path])
marks = ["10:00:20", "10:01:00", "10:03:00", "10:10:00", "10:30:00", "11:00:00", "11:06:00", "11:07:00", "11:08:00", "11:30:00"]
mi = 0
async def run():
    global mi
    async for ev in src.events(0, 2**62):
        if ev.is_book and ev.book is not None:
            m = calc.on_book(ev.leg - 1, ev.book)
            if m is None: continue
            stamp = ms_to_msk(ev.time_ms).strftime("%H:%M:%S")
            while mi < len(marks) and stamp >= marks[mi]:
                print(f"{stamp} beta={calc.beta:7.3f} beta_slow_obs={calc.beta_observations:5d} "
                      f"v1_mean={calc._mean_v1:9.3f} v2_mean={calc._mean_v2:9.3f} "
                      f"mu={m.mean_rub:9.3f} sig={m.variance**0.5:8.3f} S={m.spread_rub:8.3f} z={m.zscore:+7.2f} ticks={m.tick_count}")
                mi += 1
asyncio.run(run())
