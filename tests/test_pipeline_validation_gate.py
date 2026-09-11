import asyncio
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from src.data import DataValidator
from src.pipeline import AnalysisPipeline


def test_invalid_market_data_stops_pipeline_before_downstream_analysis():
    data = pd.DataFrame(
        {
            "open_price": [100.0, 100.0],
            "high_price": [101.0, 99.0],
            "low_price": [99.0, 98.0],
            "close_price": [100.0, 102.0],
            "volume": [1000, 1000],
        },
        index=pd.to_datetime(["2026-09-01", "2026-09-02"]),
    )

    pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
    pipeline.config = SimpleNamespace(export_config=lambda: {})
    pipeline.data_ingestion = SimpleNamespace(collect_data=AsyncMock(return_value=data))
    pipeline.data_validator = DataValidator()

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(
            pipeline.run_full_analysis(
                datetime(2026, 9, 1),
                datetime(2026, 9, 2),
                save_results=False,
                skip_storage=True,
            )
        )

    failure = json.loads(str(exc_info.value))
    assert failure["code"] == "DATA_VALIDATION_FAILED"
    assert failure["summary"]["issues_by_severity"]["error"] >= 1
    assert any(issue["rule_name"] == "invalid_high_price" for issue in failure["issues"])
