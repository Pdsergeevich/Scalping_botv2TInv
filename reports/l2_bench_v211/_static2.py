import sys
sys.path.insert(0, ".")
from pathlib import Path
from bot_a.config import AppConfig
from bot_a.economics.fees import Tariff
from scripts.l2_benchmark import _run_pair, VARIANT_OVERRIDES

cfg = AppConfig.load("appsettings.json").with_overrides(tariff=Tariff("Premium"))
base = dict(VARIANT_OVERRIDES["v210"])
betas = {"VTBR_SBER": 5.252, "MTLR_SBER": 6.451, "IRAO_SBER": 1.092}
total = 0.0
for pair, beta in betas.items():
    for label, extra in (("как в live", {}), ("beta static", {"beta_mode": "static", "beta": beta})):
        p = dict(base); p.update(extra)
        run = _run_pair(cfg, pair, ["2026-11-05"], "v210", Path("recordings"), False, p)
        net = sum(r.get("net", 0.0) for r in run["days"])
        if label == "beta static":
            total += net
        tr = " | ".join(f"{t['side']} {t['net']:+.2f} {t['reason']}" for t in run["trades"])
        print(f"{pair:<11}{label:<12} net {net:8.2f}  {tr or '— сделок нет'}")
print(f"портфель при корректной бете: {total:+.2f} RUB")
