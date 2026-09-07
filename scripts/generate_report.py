#!/usr/bin/env python3
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd

from src import AnalysisPipeline, SystemConfig
from src.analysis.valuation import (
    apply_point_in_time_valuation,
    calculate_historical_per,
    fetch_jgb_yield_history,
    fetch_nikkei_data,
)


def _format_observed_at(value) -> str:
    if pd.isna(value):
        return "Unavailable"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _status_color(status: str) -> str:
    if "Under" in status:
        return "#22c55e"
    if "Over" in status:
        return "#ef4444"
    if status == "Unavailable":
        return "#94a3b8"
    return "#eab308"


async def generate_report():
    print("Starting Report Generation...")
    config = SystemConfig()

    print("Fetching Valuation Data...")
    years = config.valuation.years_for_analysis
    premium = config.valuation.risk_premium

    price_data = fetch_nikkei_data(years)
    per_frame = calculate_historical_per(
        price_data,
        eps_provider=config.valuation.get_eps_for_date,
    )
    jgb_history = fetch_jgb_yield_history(
        config.valuation.jgb_ticker,
        per_frame.index.min(),
        per_frame.index.max(),
    )
    valuation = apply_point_in_time_valuation(
        per_frame,
        jgb_history,
        risk_premium=premium,
        per_column="estimated_per",
    )

    valuation_rows = []
    for date, row in valuation.iloc[::-1].iterrows():
        status = str(row["valuation_status"])
        color = _status_color(status)
        observed_at = _format_observed_at(row["jgb_observed_at"])
        fair_per = "Unavailable" if pd.isna(row["fair_per"]) else f"{row['fair_per']:.1f}x"
        divergence = "Unavailable" if pd.isna(row["divergence"]) else f"{row['divergence']:+.1f}%"
        valuation_rows.append(
            f"<tr><td>{date.strftime('%Y-%m')}</td><td>{row['price']:,.0f}</td>"
            f"<td>{row['estimated_per']:.1f}x</td><td>{observed_at}</td><td>{fair_per}</td>"
            f"<td>{divergence}</td><td style='color:{color}'>{status}</td></tr>"
        )

    cur = valuation.iloc[-1]
    cur_date = pd.Timestamp(cur.name)
    cur_observed_at = _format_observed_at(cur["jgb_observed_at"])
    cur_per = float(cur["estimated_per"])
    cur_fair_per = float(cur["fair_per"])
    cur_div = float(cur["divergence"])
    observation_age_days = (datetime.now().date() - cur_date.date()).days
    evidence_state = "CURRENT SNAPSHOT" if observation_age_days <= 45 else "HISTORICAL SNAPSHOT — VERIFY FIRST"
    evidence_detail = (
        "Latest market observation is recent enough for this generated report."
        if observation_age_days <= 45
        else f"Latest market observation is {observation_age_days} days old. Do not treat this as a current market signal."
    )

    print("Running Seasonality Analysis...")
    config.output_dir = Path("docs")
    config.output_dir.mkdir(exist_ok=True)

    pipeline = AnalysisPipeline(config)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10 * 365)

    results = await pipeline.run_full_analysis(start_date, end_date, save_results=True, skip_storage=True)

    seasonality_html = ""
    if results["success"]:
        viz = results.get("visualization", {})
        sig_months = results["summary"]["key_findings"].get("significant_months", [])

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if sig_months:
            sig_months_str = "".join([f"<span class='tag'>{month_names[m - 1]}</span> " for m in sorted(sig_months)])
        else:
            sig_months_str = "<span style='color:#94a3b8'>None detected (10y)</span>"

        seasonality_html = f"""
        <div class="section">
            <h2>🌸 Advanced Seasonality & Valuation Analysis</h2>

            <div class="m" style="text-align:left; margin-bottom: 2rem; display: flex; align-items: center; gap: 1rem;">
                <span class="l">Significant Months (10y):</span>
                <div>{sig_months_str}</div>
            </div>

            <div class="chart-full">
                <h3>1. Valuation Model & Deviation (Yield Gap)</h3>
                <img src="{viz.get('valuation_timeseries', '')}" alt="Valuation Time Series" loading="lazy">
                <p class="l" style="margin-top: 1rem; text-align: left;">
                    <strong>Methodology:</strong> Historical valuation uses the latest 10-year JGB observation available on or before each market observation date plus the configured risk premium. Future JGB observations are never backfilled into older rows.
                </p>
            </div>

            <div class="chart-full">
                <h3>2. Seasonal Stability Matrix (Year x Month)</h3>
                <img src="{viz.get('heatmap_year_month', '')}" alt="Year-Month Heatmap" loading="lazy">
                <p class="l" style="margin-top: 1rem; text-align: left;">
                    <strong>Insight:</strong> Detailed breakdown of monthly returns by year. Helps identify whether a seasonal pattern is consistent over time or driven by outliers.
                </p>
            </div>

            <div class="chart-full">
                <h3>3. Monthly Return Distributions</h3>
                <img src="{viz.get('boxplot_distribution', '')}" alt="Monthly Boxplots" loading="lazy">
                <p class="l" style="margin-top: 1rem; text-align: left;">
                    <strong>Statistical Robustness:</strong> Boxplots show the median, interquartile range (IQR), and outliers for each month. Narrow boxes indicate consistent behavior; wide boxes indicate high volatility.
                </p>
            </div>

            <div class="grid-2">
                <div class="chart-box">
                    <h3>Mean Returns (Avg)</h3>
                    <img src="{viz.get('seasonality_returns_chart', '')}" alt="Returns Bar Chart" loading="lazy">
                </div>
                <div class="chart-box">
                    <h3>Statistical Significance (P-Values)</h3>
                    <img src="{viz.get('seasonality_pvalues_chart', '')}" alt="Significance Chart" loading="lazy">
                </div>
            </div>

            <div class="chart-full">
                <h3>5. Cumulative Seasonality Trend</h3>
                <img src="{viz.get('timeseries_plot', '')}" alt="Seasonality Time Series" loading="lazy">
                <p class="l" style="margin-top: 1rem; text-align: left;">
                    <strong>Test Specification:</strong> Seasonality evaluated via One-Sample T-test (Null: Mean=0). Significance threshold p &lt; 0.05 (Bonferroni corrected).
                </p>
            </div>
        </div>
        """
    else:
        seasonality_html = "<p class='error'>Seasonality Analysis Failed.</p>"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NK225 Valuation & Seasonality</title>
    <style>
        :root {{
            --bg-body: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --accent-yellow: #eab308;
            --border-color: #334155;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-body);
            color: var(--text-primary);
            padding: 2rem;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{ font-size: 2rem; font-weight: 800; letter-spacing: -0.025em; margin-bottom: 0.5rem; background: linear-gradient(to right, #fff, #cbd5e1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        h2 {{ font-size: 1.5rem; font-weight: 700; margin: 3rem 0 1.5rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border-color); display: flex; align-items: center; gap: 0.5rem; }}
        h3 {{ color: var(--text-secondary); font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1rem; }}
        p.date {{ color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 1rem; }}
        .evidence {{ background: var(--bg-card); border: 1px solid var(--border-color); border-left: 4px solid var(--accent-yellow); border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 2rem; }}
        .evidence strong {{ display: block; margin-bottom: 0.25rem; }}
        .evidence span {{ color: var(--text-secondary); font-size: 0.875rem; }}
        .g {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
        .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
        .m {{ background: var(--bg-card); padding: 1.5rem; border-radius: 16px; border: 1px solid var(--border-color); text-align: center; transition: transform 0.2s, box-shadow 0.2s; }}
        .m:hover {{ transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); border-color: #475569; }}
        .v {{ font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; letter-spacing: -0.025em; }}
        .l {{ color: var(--text-secondary); font-size: 0.875rem; font-weight: 500; }}
        .table-container {{ overflow-x: auto; border-radius: 16px; border: 1px solid var(--border-color); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }}
        table {{ width: 100%; border-collapse: collapse; white-space: nowrap; background: var(--bg-card); }}
        th, td {{ padding: 1rem 1.5rem; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ color: var(--text-secondary); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; background: #0f172a; }}
        tr:last-child td {{ border-bottom: none; }}
        td {{ font-variant-numeric: tabular-nums; }}
        .chart-box, .chart-full {{ background: var(--bg-card); padding: 1.5rem; border-radius: 16px; border: 1px solid var(--border-color); }}
        img {{ width: 100%; height: auto; border-radius: 8px; mix-blend-mode: screen; filter: contrast(1.1); }}
        .u {{ color: var(--accent-green); }}
        .o {{ color: var(--accent-red); }}
        .w {{ color: var(--accent-yellow); }}
        .tag {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; background: rgba(34, 197, 94, 0.2); color: #4ade80; font-size: 0.875rem; font-weight: 600; }}
        @media (max-width: 768px) {{
            body {{ padding: 1rem; }}
            .v {{ font-size: 1.5rem; }}
            .grid-2 {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <h1>NK225 Market Intelligence</h1>
    <p class="date">Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M JST')}</p>
    <div class="evidence" role="status" aria-label="Valuation evidence status">
        <strong>{evidence_state}</strong>
        <span>Market data through {cur_date.strftime('%Y-%m-%d')} · JGB observation {cur_observed_at}. {evidence_detail}</span>
    </div>

    <div class="section">
        <h2>📊 Market Valuation</h2>
        <div class="g">
            <div class="m"><div class="v">{cur['price']:,.0f}</div><div class="l">Latest observed price</div></div>
            <div class="m"><div class="v">{cur_per:.1f}x</div><div class="l">Observed PER</div></div>
            <div class="m"><div class="v">{cur_fair_per:.1f}x</div><div class="l">Point-in-time fair PER</div></div>
            <div class="m"><div class="v {'o' if cur_div > 0 else 'u'}">{cur_div:+.1f}%</div><div class="l">Point-in-time divergence</div></div>
        </div>
        <div class="table-container">
            <table>
                <tr><th>Date</th><th>Price</th><th>PER</th><th>JGB observed</th><th>Fair PER</th><th>Divergence</th><th>Status</th></tr>
                {''.join(valuation_rows)}
            </table>
        </div>
    </div>

    {seasonality_html}

</body>
</html>
"""

    output_path = Path("docs/index.html")
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Report generated at: {output_path.absolute()}")


if __name__ == "__main__":
    asyncio.run(generate_report())
