import sys, asyncio
sys.path.insert(0, ".")
from pathlib import Path
from bot_a.backtest.tick_replay import JsonlRecordingSource
from bot_a.config import AppConfig
from bot_a.math_engine.pair_spread import PairSpreadCalculator
from bot_a.utils.time_provider import ms_to_msk

cfg = AppConfig.load("appsettings.json").with_preset_overrides()
preset = cfg.preset("VTBR_SBER")
s = cfg.strategy
calc = PairSpreadCalculator(
    leg1=preset.leg1, leg2=preset.leg2, beta=s.beta,
    ewma_window=s.ewma_window, obi_depth=s.obi_depth, beta_mode=s.beta_mode,
    beta_min=s.beta_min, beta_max=s.beta_max, beta_min_samples=s.beta_min_samples,
    macro_beta_window=s.macro_beta_window, macro_beta_min_samples=s.macro_beta_min_samples,
    stat_sample_interval_seconds=s.stat_sample_interval_seconds,
    stat_freeze_until_msk=s.stat_freeze_until_msk, stat_freeze_windows=s.stat_freeze_windows,
    max_valid_spread_bps=s.max_valid_spread_bps,
)
src = JsonlRecordingSource([Path("recordings/VTBR_SBER_2026-11-05.jsonl.gz")])
leg_calls = {0: 0, 1: 0}
gaps = []
last = {0: None, 1: None}
async def run():
    async for ev in src.events(0, 2**62):
        if ev.is_book and ev.book is not None:
            leg = ev.leg - 1
            leg_calls[leg] += 1
            if last[leg] is not None:
                gaps.append((ev.time_ms - last[leg], leg))
            last[leg] = ev.time_ms
            calc.on_book(leg, ev.book)
asyncio.run(run())
print("вызовов по ногам", leg_calls, "всего", sum(leg_calls.values()))
print("beta_observations", calc.beta_observations, "ewma ticks", calc.ewma.count)
import statistics
for leg in (0, 1):
    vals = [g for g, l in gaps if l == leg]
    if vals:
        print(f"leg{leg+1} gap ms: медиана {statistics.median(vals):.0f}, макс {max(vals)}, мин {min(vals)}")
