from datetime import datetime
from typing import Any, Dict, Optional

from .analysis import MechanismAnalyzer, SeasonalityAnalyzer, SeasonalRegressionModel
from .config import SystemConfig
from .data import DataIngestionPipeline, DataValidator, MarketDataRepository
from .options import OptionsCalculator
from .risk import MonteCarloEngine, VaRCalculator
from .visualization import OptionsVisualizer, RiskVisualizer, SeasonalityVisualizer


class AnalysisPipeline:
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()

        self.data_ingestion = DataIngestionPipeline(self.config)
        self.data_validator = DataValidator()
        self.data_repository = MarketDataRepository(self.config)
        self.options_calculator = OptionsCalculator()
        self.mc_engine = MonteCarloEngine(self.config.analysis.monte_carlo_simulations)
        self.var_calculator = VaRCalculator()

        self.seasonality_viz = SeasonalityVisualizer(self.config.output_dir)
        self.options_viz = OptionsVisualizer(self.config.output_dir)
        self.risk_viz = RiskVisualizer(self.config.output_dir)

    async def run_full_analysis(
        self, start_date: datetime, end_date: datetime, save_results: bool = True
    ) -> Dict[str, Any]:
        pipeline_results = {
            "metadata": {
                "start_date": start_date,
                "end_date": end_date,
                "analysis_timestamp": datetime.now(),
                "config_used": self.config.export_config(),
            }
        }

        # Phase 1: Data
        raw_data = await self.data_ingestion.collect_data(start_date, end_date)
        validation_results = self.data_validator.validate_dataset(raw_data)
        self.data_repository.store_data(raw_data, "pipeline_analysis")

        pipeline_results["data_phase"] = {
            "success": True,
            "validated_data": raw_data,
            "data_quality_valid": validation_results.is_valid,
        }

        # Phase 2: Analysis
        data = raw_data
        seasonality_analyzer = SeasonalityAnalyzer(data, self.config.analysis.significance_level)
        seasonality_results = seasonality_analyzer.test_monthly_patterns()
        dow_results = seasonality_analyzer.test_day_of_week_patterns()
        quarter_results = seasonality_analyzer.test_quarter_patterns()
        rolling_results = seasonality_analyzer.rolling_seasonality_analysis(5, 6)

        regression_model = SeasonalRegressionModel(data)
        regression_model.fit_seasonal_model()

        mechanism_analyzer = MechanismAnalyzer(data)
        mechanism_results = mechanism_analyzer.comprehensive_mechanism_analysis()

        pipeline_results["analysis_phase"] = {
            "success": True,
            "seasonality_results": seasonality_results,
            "day_of_week_results": dow_results,
            "quarterly_results": quarter_results,
            "rolling_analysis": rolling_results,
            "mechanism_analysis": mechanism_results,
            "significant_months": [m for m, r in seasonality_results.items() if r.is_significant],
        }

        # Phase 6: Summary
        pipeline_results["summary"] = {
            "key_findings": {"significant_months": pipeline_results["analysis_phase"]["significant_months"]}
        }

        pipeline_results["success"] = True
        return pipeline_results
