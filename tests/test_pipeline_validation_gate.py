import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd

from src.data.validation import DataValidator
from src.pipeline import AnalysisPipeline


class _Config:
    analysis = SimpleNamespace(significance_level=0.05)
    valuation = SimpleNamespace(
        jgb_ticker="TEST-JGB",
        risk_premium=0.03,
        get_eps_for_date=lambda _date: 2_000.0,
    )

    @staticmethod
    def export_config():
        return {}


def _pipeline(frame: pd.DataFrame) -> AnalysisPipeline:
    pipeline = object.__new__(AnalysisPipeline)
    pipeline.config = _Config()
    pipeline.data_ingestion = SimpleNamespace(collect_data=AsyncMock(return_value=frame.copy()))
    pipeline.data_validator = DataValidator()
    pipeline.data_repository = Mock()
    return pipeline


def _valid_frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-05", periods=40, freq="B")
    close = pd.Series(range(40), index=index, dtype="float64") + 30_000.0
    return pd.DataFrame(
        {
            "open_price": close,
            "high_price": close + 10.0,
            "low_price": close - 10.0,
            "close_price": close,
            "adjusted_close": close,
            "volume": 1_000_000.0,
        },
        index=index,
    )


def _run(pipeline: AnalysisPipeline):
    return asyncio.run(
        pipeline.run_full_analysis(
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-03-01"),
            save_results=False,
            skip_storage=True,
        )
    )


def test_missing_ohlc_column_fails_before_analysis_or_valuation():
    frame = _valid_frame().drop(columns=["high_price"])
    pipeline = _pipeline(frame)

    with (
        patch("src.pipeline.fetch_jgb_yield_history") as fetch_jgb,
        patch("src.pipeline.SeasonalityAnalyzer") as seasonality,
    ):
        result = _run(pipeline)

    assert result["success"] is False
    assert result["data_phase"]["success"] is False
    assert result["data_phase"]["validation_failure"]["reason"] == "data_validation_failed"
    assert "missing_column" in {
        issue["rule_name"] for issue in result["data_phase"]["validation_failure"]["blocking_issues"]
    }
    fetch_jgb.assert_not_called()
    seasonality.assert_not_called()
    pipeline.data_repository.store_data.assert_not_called()


def test_non_positive_price_failure_is_deterministic_and_stops_analysis():
    frame = _valid_frame()
    frame.loc[frame.index[3], "low_price"] = 0.0

    with patch("src.pipeline.SeasonalityAnalyzer") as seasonality:
        first = _run(_pipeline(frame))
        second = _run(_pipeline(frame))

    assert first["success"] is False
    assert first["data_phase"]["validation_failure"] == second["data_phase"]["validation_failure"]
    assert "non_positive_price" in {
        issue["rule_name"] for issue in first["data_phase"]["validation_failure"]["blocking_issues"]
    }
    seasonality.assert_not_called()


def test_valid_data_preserves_analysis_completion():
    frame = _valid_frame()
    pipeline = _pipeline(frame)
    jgb_index = pd.DatetimeIndex([frame.index[0]])
    jgb_history = pd.Series([1.0], index=jgb_index)

    seasonality = Mock()
    seasonality.test_monthly_patterns.return_value = {}
    seasonality.test_day_of_week_patterns.return_value = {}
    seasonality.test_quarter_patterns.return_value = {}
    seasonality.rolling_seasonality_analysis.return_value = {}

    def add_valuation_columns(data, *_args, **_kwargs):
        result = data.copy()
        result["jgb_yield"] = 1.0
        return result

    with (
        patch("src.pipeline.fetch_jgb_yield_history", return_value=jgb_history),
        patch("src.pipeline.fetch_current_jgb_yield", return_value=1.0),
        patch("src.pipeline.apply_point_in_time_valuation", side_effect=add_valuation_columns),
        patch("src.pipeline.SeasonalityAnalyzer", return_value=seasonality) as seasonality_cls,
        patch("src.pipeline.SeasonalRegressionModel") as regression_cls,
        patch("src.pipeline.MechanismAnalyzer") as mechanism_cls,
    ):
        mechanism_cls.return_value.comprehensive_mechanism_analysis.return_value = {}
        result = _run(pipeline)

    assert result["success"] is True
    assert result["data_phase"]["data_quality_valid"] is True
    assert result["analysis_phase"]["success"] is True
    seasonality_cls.assert_called_once()
    regression_cls.return_value.fit_seasonal_model.assert_called_once_with()
    mechanism_cls.return_value.comprehensive_mechanism_analysis.assert_called_once_with()
