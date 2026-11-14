import pandas as pd

from src.analysis.seasonality import SeasonalityAnalyzer


def test_rolling_seasonality_with_no_complete_window_is_unavailable():
    dates = pd.date_range("2025-01-01", periods=30, freq="B")
    data = pd.DataFrame({"returns": [0.01] * len(dates)}, index=dates)

    result = SeasonalityAnalyzer(data).rolling_seasonality_analysis(window_years=5)

    assert result["rolling_windows"] == []
    consistency = result["consistency_analysis"]
    assert consistency["total_windows"] == 0
    assert consistency["consistent_months"] == {}
    assert set(consistency["month_significance_frequency"].values()) == {None}
