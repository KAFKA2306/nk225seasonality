from __future__ import annotations

import json

import pytest

from src.analysis.browser_scenario import METHOD, calculate_scenario, calculate_scenario_json
from src.analysis.valuation import ValuationAnalyzer, ValuationMetrics


def test_browser_scenario_matches_existing_python_valuation_formula() -> None:
    current_per = 18.5
    jgb_yield = 1.35
    risk_premium = 5.0

    browser = calculate_scenario(
        current_per=current_per,
        jgb_yield=jgb_yield,
        risk_premium=risk_premium,
    )
    existing = ValuationAnalyzer().calculate_valuation_status(
        ValuationMetrics(
            jgb_yield=jgb_yield,
            current_per=current_per,
            risk_premium=risk_premium,
        )
    )

    assert browser["fair_per"] == pytest.approx(existing["fair_per"], abs=0.01)
    assert browser["earnings_yield"] == pytest.approx(existing["earnings_yield"], abs=0.01)
    assert browser["yield_gap"] == pytest.approx(existing["yield_gap"], abs=0.01)
    assert browser["divergence_pct"] == pytest.approx(existing["divergence_pct"], abs=0.01)
    assert browser["canonical_point_in_time_overwritten"] is False
    assert browser["method"] == METHOD


def test_browser_scenario_json_is_deterministic() -> None:
    payload = json.dumps({"current_per": 20, "jgb_yield": 1.5, "risk_premium": 5.0})
    assert calculate_scenario_json(payload) == calculate_scenario_json(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current_per": 0, "jgb_yield": 1.0, "risk_premium": 5.0},
        {"current_per": 20, "jgb_yield": -6.0, "risk_premium": 5.0},
        {"current_per": float("nan"), "jgb_yield": 1.0, "risk_premium": 5.0},
    ],
)
def test_browser_scenario_fails_closed_for_invalid_inputs(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        calculate_scenario(**kwargs)
