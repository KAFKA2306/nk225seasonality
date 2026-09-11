from datetime import datetime
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

import pandas as pd

from src.data.validation import DataValidator
from src.pipeline import AnalysisPipeline, DataValidationError


class PipelineValidationGateTest(IsolatedAsyncioTestCase):
    async def test_invalid_market_data_stops_before_storage_and_analysis(self):
        invalid_data = pd.DataFrame(
            {
                "open_price": [100.0, 101.0],
                "low_price": [99.0, 100.0],
                "close_price": [100.5, 101.5],
                "volume": [1000, 1100],
            },
            index=pd.to_datetime(["2026-01-05", "2026-01-06"]),
        )

        pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
        pipeline.config = SimpleNamespace(export_config=lambda: {})
        pipeline.data_ingestion = SimpleNamespace(
            collect_data=AsyncMock(return_value=invalid_data)
        )
        pipeline.data_validator = DataValidator()
        pipeline.data_repository = SimpleNamespace(store_data=Mock())

        with self.assertRaises(DataValidationError) as caught:
            await pipeline.run_full_analysis(
                datetime(2026, 1, 5),
                datetime(2026, 1, 6),
                save_results=False,
            )

        self.assertFalse(caught.exception.validation_result.is_valid)
        self.assertIn("missing_column", str(caught.exception))
        pipeline.data_repository.store_data.assert_not_called()

    async def test_non_positive_price_is_rejected_deterministically(self):
        invalid_data = pd.DataFrame(
            {
                "open_price": [100.0],
                "high_price": [101.0],
                "low_price": [0.0],
                "close_price": [100.5],
                "volume": [1000],
            },
            index=pd.to_datetime(["2026-01-05"]),
        )

        pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
        pipeline.config = SimpleNamespace(export_config=lambda: {})
        pipeline.data_ingestion = SimpleNamespace(
            collect_data=AsyncMock(return_value=invalid_data)
        )
        pipeline.data_validator = DataValidator()
        pipeline.data_repository = SimpleNamespace(store_data=Mock())

        for _ in range(2):
            with self.assertRaisesRegex(
                DataValidationError,
                "Market data validation failed: non_positive_price",
            ):
                await pipeline.run_full_analysis(
                    datetime(2026, 1, 5),
                    datetime(2026, 1, 5),
                    save_results=False,
                )

        pipeline.data_repository.store_data.assert_not_called()
