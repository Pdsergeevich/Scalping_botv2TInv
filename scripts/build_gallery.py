"""Сборка страницы-галереи по всем наборам графиков в reports/charts.

Использование:  py -3 scripts/build_gallery.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "reports" / "charts"

# Порядок карточек: сначала самые содержательные прогоны.
ORDER = {
    "charts_vtbr_6m": 0,
    "paper_vtbr_oct": 1,
    "replay_l2_prod": 2,
    "replay_l2_demo": 3,
    "paper_live_vtbr_1026": 4,
    "paper_vtbr_today": 5,
}

TEMPLATE = """<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<title>Bot A — галерея прогонов</title><style>
body{{background:#0f1419;color:#d7e0ea;font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:26px}}
h1{{font-size:21px;margin:0 0 6px}}.muted{{color:#8b9aab;font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:18px;margin-top:18px}}
.card{{display:block;background:#161d26;border:1px solid #243040;border-radius:10px;padding:12px;
text-decoration:none;color:inherit}}.card:hover{{border-color:#4fa3ff}}
.title{{font-size:15px;font-weight:600;margin-bottom:4px}}.row{{font-size:12px;margin:2px 0}}
img{{width:100%;border-radius:6px;margin-top:8px;border:1px solid #243040}}</style></head><body>
<h1>Bot A — прогоны на реальных данных MOEX</h1>
<div class="muted">Клик по карточке — подробный отчёт: графики, Z-score с точками входов/выходов,
таблица всех сделок (когда вошёл, когда вышел, цена, причина, net).</div>
<div class="grid">{cards}</div></body></html>
"""


def build() -> Path:
    directories = [p for p in ROOT.iterdir() if p.is_dir() and (p / "charts.json").exists()]
    directories.sort(key=lambda p: (ORDER.get(p.name, 99), p.name))
    cards = []
    for directory in directories:
        meta = json.loads((directory / "charts.json").read_text(encoding="utf-8"))
        net = float(meta["netPnlRub"])
        color = "#2ecc71" if net >= 0 else "#ff5c5c"
        cards.append(
            f'<a class="card" href="{directory.name}/index.html">'
            f'<div class="title">{directory.name}</div>'
            f'<div class="row">{meta["preset"]} · {meta["topology"]}</div>'
            f'<div class="row">сделок <b>{meta["trades"]}</b> · net '
            f'<b style="color:{color}">{net:+.2f} RUB</b></div>'
            f'<div class="row muted">gross {float(meta["grossPnlRub"]):+.2f} · '
            f'fees {float(meta["feesRub"]):.2f}</div>'
            f'<img src="{directory.name}/01_overview.png" alt="overview"></a>'
        )
    path = ROOT / "index.html"
    path.write_text(TEMPLATE.format(cards="".join(cards)), encoding="utf-8")
    return path


if __name__ == "__main__":
    print("галерея:", build())
