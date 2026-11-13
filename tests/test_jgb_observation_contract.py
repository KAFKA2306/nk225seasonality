import pandas as pd
import pytest

from src.analysis.valuation import apply_point_in_time_valuation


@pytest.mark.parametrize(
    "observed_at",
    [
        ["2026-01-01", "2026-01-01"],
        ["2026-01-01", None],
    ],
)
def test_ambiguous_jgb_observation_dates_fail_closed(observed_at):
    prices = pd.DataFrame(
        {"estimated_per": [20.0]}, index=pd.to_datetime(["2026-01-02"])
    )
    jgb = pd.DataFrame(
        {"jgb_yield": [1.0, 2.0]}, index=pd.to_datetime(observed_at)
    )
    with pytest.raises(ValueError, match="timestamps must be present and unique"):
        apply_point_in_time_valuation(
            prices, jgb, risk_premium=3.5, per_column="estimated_per"
        )
