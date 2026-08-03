from __future__ import annotations

import math

import pandas as pd

from src.analysis.valuation import apply_point_in_time_valuation


def valuation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"estimated_per": [20.0, 20.0, 20.0]},
        index=pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"]),
    )


def yield_history() -> pd.DataFrame:
    frame = pd.DataFrame(
        {"jgb_yield": [0.50, 1.00, 1.25]},
        index=pd.to_datetime(["2025-01-15", "2025-02-15", "2025-04-15"]),
    )
    frame.index.name = "jgb_observed_at"
    return frame


def test_each_month_uses_latest_jgb_observation_at_or_before_date() -> None:
    result = apply_point_in_time_valuation(
        valuation_frame(),
        yield_history(),
        risk_premium=3.5,
        per_column="estimated_per",
    )

    assert result["jgb_yield"].tolist() == [0.50, 1.00, 1.00]
    assert result["jgb_observed_at"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-01-15",
        "2025-02-15",
        "2025-02-15",
    ]
    assert result.loc[pd.Timestamp("2025-01-31"), "fair_per"] == 25.0
    assert math.isclose(
        result.loc[pd.Timestamp("2025-02-28"), "fair_per"],
        100 / 4.5,
    )


def test_future_jgb_observation_is_never_backfilled() -> None:
    frame = pd.DataFrame(
        {"estimated_per": [18.0]},
        index=pd.to_datetime(["2025-01-01"]),
    )
    future_only = pd.DataFrame(
        {"jgb_yield": [0.75]},
        index=pd.to_datetime(["2025-01-02"]),
    )

    result = apply_point_in_time_valuation(
        frame,
        future_only,
        risk_premium=3.5,
        per_column="estimated_per",
    )

    assert pd.isna(result.iloc[0]["jgb_yield"])
    assert pd.isna(result.iloc[0]["jgb_observed_at"])
    assert pd.isna(result.iloc[0]["fair_per"])
    assert result.iloc[0]["valuation_status"] == "Unavailable"


def test_current_rate_revaluation_is_separate_from_historical_series() -> None:
    historical = apply_point_in_time_valuation(
        valuation_frame(),
        yield_history(),
        risk_premium=3.5,
        per_column="estimated_per",
        current_jgb_yield=2.0,
    )
    changed_current = apply_point_in_time_valuation(
        valuation_frame(),
        yield_history(),
        risk_premium=3.5,
        per_column="estimated_per",
        current_jgb_yield=3.0,
    )

    pd.testing.assert_series_equal(historical["fair_per"], changed_current["fair_per"])
    pd.testing.assert_series_equal(historical["divergence"], changed_current["divergence"])
    assert not historical["current_rate_revaluation_fair_per"].equals(
        changed_current["current_rate_revaluation_fair_per"]
    )


def test_timezone_aware_inputs_join_without_date_drift() -> None:
    frame = pd.DataFrame(
        {"estimated_per": [19.0]},
        index=pd.DatetimeIndex(["2025-03-31T15:00:00+09:00"]),
    )
    history = pd.DataFrame(
        {"jgb_yield": [1.1]},
        index=pd.DatetimeIndex(["2025-03-31T05:00:00+00:00"]),
    )

    result = apply_point_in_time_valuation(
        frame,
        history,
        risk_premium=3.5,
        per_column="estimated_per",
    )

    assert result.iloc[0]["jgb_yield"] == 1.1
    assert result.iloc[0]["valuation_method"] == "historical_point_in_time_jgb_asof"
