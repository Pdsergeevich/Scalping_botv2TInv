import sys, logging, io
sys.path.insert(0, ".")
from pathlib import Path
from bot_a.config import AppConfig
from bot_a.economics.fees import Tariff
from scripts.l2_benchmark import _run_pair, VARIANT_OVERRIDES

cfg = AppConfig.load("appsettings.json").with_overrides(tariff=Tariff("Premium"))
for pair in ("VTBR_SBER", "MTLR_SBER", "IRAO_SBER"):
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    run = _run_pair(cfg, pair, ["2026-11-05"], "v210", Path("recordings"), False, dict(VARIANT_OVERRIDES["v210"]))
    root.removeHandler(handler)
    lines = [l for l in buf.getvalue().splitlines() if "Анти-нож" in l and "пауза" not in l]
    net = sum(r.get("net", 0.0) for r in run["days"])
    print(f"== {pair}: net {net:+.2f}, сработок анти-ножа {len(lines)}")
    for l in lines[:6]:
        print("   ", l.split("bot_a: ")[-1][:130])
