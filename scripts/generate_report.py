#!/usr/bin/env python3
import sys

sys.path.insert(0, ".")
from datetime import datetime

import yfinance as yf

years, jgb, premium, eps = 5, 3.5, 3.5, 2400
df = yf.Ticker("^N225").history(period=f"{years}y", interval="1mo")
fair_per = 1 / ((jgb + premium) / 100)

rows = []
for date, row in df.iterrows():
    per = row["Close"] / eps
    div = ((per - fair_per) / fair_per) * 100
    status = (
        "Significantly Overvalued"
        if div > 20
        else "Overvalued"
        if div > 10
        else "Significantly Undervalued"
        if div < -20
        else "Undervalued"
        if div < -10
        else "Fairly Valued"
    )
    color = "#22c55e" if "Under" in status else "#ef4444" if "Over" in status else "#eab308"
    rows.append(
        f"<tr><td>{date.strftime('%Y-%m')}</td><td>{row['Close']:,.0f}</td><td>{per:.1f}x</td><td>{div:+.1f}%</td><td style='color:{color}'>{status}</td></tr>"
    )

cur = df.iloc[-1]
cur_per, cur_div = (
    cur["Close"] / eps,
    ((cur["Close"] / eps - fair_per) / fair_per) * 100,
)

print(f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>NK225 Valuation</title>
<style>*{{margin:0;padding:0}}body{{font-family:system-ui;background:#0f172a;color:#e2e8f0;padding:2rem}}h1{{color:#f8fafc}}
.g{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1.5rem 0}}.m{{background:#1e293b;padding:1rem;border-radius:8px;text-align:center}}
.v{{font-size:1.5rem;font-weight:bold}}.l{{color:#94a3b8;font-size:.75rem}}table{{width:100%;border-collapse:collapse;background:#1e293b;border-radius:8px}}
th,td{{padding:.5rem;text-align:left;border-bottom:1px solid #334155}}th{{color:#94a3b8}}.u{{color:#22c55e}}.o{{color:#ef4444}}</style></head>
<body><h1>NK225 Valuation</h1><p style="color:#64748b">{datetime.now().strftime("%Y-%m-%d")}</p>
<div class="g"><div class="m"><div class="v">{cur["Close"]:,.0f}</div><div class="l">Price</div></div>
<div class="m"><div class="v">{cur_per:.1f}x</div><div class="l">PER</div></div>
<div class="m"><div class="v">{fair_per:.1f}x</div><div class="l">Fair PER</div></div>
<div class="m"><div class="v {"o" if cur_div > 0 else "u"}">{cur_div:+.1f}%</div><div class="l">Divergence</div></div></div>
<table><tr><th>Date</th><th>Price</th><th>PER</th><th>Div</th><th>Status</th></tr>{"".join(rows)}</table></body></html>""")
