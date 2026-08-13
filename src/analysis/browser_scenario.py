from __future__ import annotations

import json
import math
from typing import Any

METHOD = "user_input_current_rate_scenario"


def calculate_scenario(*, current_per: float, jgb_yield: float, risk_premium: float) -> dict[str, Any]:
    """Calculate a non-canonical current-rate valuation scenario.

    This function is intentionally dependency-free so the same Python source can
    execute in CPython tests and Pyodide without loading pandas/numpy/yfinance.
    It never fetches market data and never mutates point-in-time observations.
    """
    values = {
        "current_per": float(current_per),
        "jgb_yield": float(jgb_yield),
        "risk_premium": float(risk_premium),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("scenario inputs must be finite numbers")
    if values["current_per"] <= 0:
        raise ValueError("current_per must be > 0")

    discount_rate = values["jgb_yield"] + values["risk_premium"]
    if discount_rate <= 0:
        raise ValueError("jgb_yield + risk_premium must be > 0")

    fair_per = 100.0 / discount_rate
    earnings_yield = 100.0 / values["current_per"]
    divergence_pct = ((values["current_per"] - fair_per) / fair_per) * 100.0

    return {
        **values,
        "earnings_yield": round(earnings_yield, 4),
        "yield_gap": round(earnings_yield - values["jgb_yield"], 4),
        "fair_per": round(fair_per, 4),
        "divergence_pct": round(divergence_pct, 4),
        "method": METHOD,
        "canonical_point_in_time_overwritten": False,
    }


def calculate_scenario_json(payload: str) -> str:
    data = json.loads(payload)
    result = calculate_scenario(
        current_per=data["current_per"],
        jgb_yield=data["jgb_yield"],
        risk_premium=data["risk_premium"],
    )
    return json.dumps(result, sort_keys=True, separators=(",", ":"))
