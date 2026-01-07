from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor

from ..config import JapaneseMarketConstants, get_logger


class MechanismType(Enum):
    """Types of mechanism factors."""

    INSTITUTIONAL = "institutional"
    REGULATORY = "regulatory"
    CULTURAL = "cultural"
    ECONOMIC = "economic"
    TECHNICAL = "technical"


@dataclass
class MechanismResult:
    """Result of mechanism analysis."""

    factor_name: str
    mechanism_type: MechanismType
    effect_size: float
    statistical_significance: float
    confidence_interval: Tuple[float, float]
    affected_periods: List[datetime]
    explanation: str


class MechanismAnalyzer:
    """Analyze structural factors driving seasonality."""

    def __init__(
        self,
        market_data: pd.DataFrame,
        external_data: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        self.market_data = market_data.copy()
        self.external_data = external_data or {}
        self.logger = get_logger(__name__)

        self._prepare_mechanism_features()

    def _prepare_mechanism_features(self):
        self.market_data["fiscal_year_end"] = (
            (self.market_data.index.month == JapaneseMarketConstants.FISCAL_YEAR_END_MONTH)
            & (self.market_data.index.day >= 20)
        ).astype(int)

        self.market_data["fiscal_year_start"] = (
            (self.market_data.index.month == JapaneseMarketConstants.FISCAL_YEAR_START_MONTH)
            & (self.market_data.index.day <= 10)
        ).astype(int)

        self.market_data["golden_week"] = self._identify_golden_week()
        self.market_data["year_end_holidays"] = self._identify_year_end_holidays()
        self.market_data["obon_period"] = self._identify_obon_period()

        self.market_data["earnings_season_q1"] = self._identify_earnings_season(1)
        self.market_data["earnings_season_q2"] = self._identify_earnings_season(2)
        self.market_data["earnings_season_q3"] = self._identify_earnings_season(3)
        self.market_data["earnings_season_q4"] = self._identify_earnings_season(4)

        self.market_data["summer_bonus"] = (self.market_data.index.month == 7).astype(int)
        self.market_data["winter_bonus"] = (self.market_data.index.month == 12).astype(int)

        self.market_data["month_end"] = (self.market_data.index.day >= 28).astype(int)
        self.market_data["month_beginning"] = (self.market_data.index.day <= 3).astype(int)

        self.market_data["pension_rebalancing"] = self._identify_pension_rebalancing()

        self.market_data["tax_loss_selling"] = (
            (self.market_data.index.month == 12) & (self.market_data.index.day >= 15)
        ).astype(int)

        self.logger.info("Mechanism features prepared")

    def _identify_golden_week(self) -> pd.Series:
        return (
            ((self.market_data.index.month == 4) & (self.market_data.index.day >= 29))
            | ((self.market_data.index.month == 5) & (self.market_data.index.day <= 5))
        ).astype(int)

    def _identify_year_end_holidays(self) -> pd.Series:
        return (
            ((self.market_data.index.month == 12) & (self.market_data.index.day >= 29))
            | ((self.market_data.index.month == 1) & (self.market_data.index.day <= 3))
        ).astype(int)

    def _identify_obon_period(self) -> pd.Series:
        return (
            (self.market_data.index.month == 8)
            & (self.market_data.index.day >= 13)
            & (self.market_data.index.day <= 16)
        ).astype(int)

    def _identify_earnings_season(self, quarter: int) -> pd.Series:
        earnings_months = {
            1: [5, 6],
            2: [8, 9],
            3: [11, 12],
            4: [2, 3],
        }

        months = earnings_months.get(quarter, [])
        return self.market_data.index.month.isin(months).astype(int)

    def _identify_pension_rebalancing(self) -> pd.Series:
        return (
            ((self.market_data.index.month == 3) & (self.market_data.index.day >= 25))
            | ((self.market_data.index.month == 6) & (self.market_data.index.day >= 25))
            | ((self.market_data.index.month == 9) & (self.market_data.index.day >= 25))
            | ((self.market_data.index.month == 12) & (self.market_data.index.day >= 25))
        ).astype(int)

    def analyze_fiscal_year_effects(self) -> MechanismResult:
        fy_end_mask = self.market_data["fiscal_year_end"] == 1

        fy_end_returns = self.market_data.loc[fy_end_mask, "returns"]
        other_returns = self.market_data.loc[~fy_end_mask, "returns"]

        t_stat, p_value = stats.ttest_ind(fy_end_returns, other_returns)

        pooled_std = np.sqrt(
            (
                (len(fy_end_returns) - 1) * fy_end_returns.std() ** 2
                + (len(other_returns) - 1) * other_returns.std() ** 2
            )
            / (len(fy_end_returns) + len(other_returns) - 2)
        )

        effect_size = (fy_end_returns.mean() - other_returns.mean()) / pooled_std

        se_diff = pooled_std * np.sqrt(1 / len(fy_end_returns) + 1 / len(other_returns))
        df = len(fy_end_returns) + len(other_returns) - 2
        t_critical = stats.t.ppf(0.975, df)

        mean_diff = fy_end_returns.mean() - other_returns.mean()
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff

        affected_dates = self.market_data[fy_end_mask].index.tolist()

        return MechanismResult(
            factor_name="fiscal_year_end",
            mechanism_type=MechanismType.INSTITUTIONAL,
            effect_size=effect_size,
            statistical_significance=p_value,
            confidence_interval=(ci_lower, ci_upper),
            affected_periods=affected_dates,
            explanation=f"Fiscal year-end effect: {fy_end_returns.mean():.4f} vs {other_returns.mean():.4f} (difference: {mean_diff:.4f})",
        )

    def analyze_holiday_effects(self) -> Dict[str, MechanismResult]:
        holiday_factors = ["golden_week", "year_end_holidays", "obon_period"]

        results = {}

        for factor in holiday_factors:
            if factor in self.market_data.columns:
                holiday_mask = self.market_data[factor] == 1

                if holiday_mask.sum() > 0:
                    holiday_returns = self.market_data.loc[holiday_mask, "returns"]
                    normal_returns = self.market_data.loc[~holiday_mask, "returns"]

                    t_stat, p_value = stats.ttest_ind(holiday_returns, normal_returns)

                    pooled_std = np.sqrt(
                        (
                            (len(holiday_returns) - 1) * holiday_returns.std() ** 2
                            + (len(normal_returns) - 1) * normal_returns.std() ** 2
                        )
                        / (len(holiday_returns) + len(normal_returns) - 2)
                    )

                    effect_size = (holiday_returns.mean() - normal_returns.mean()) / pooled_std

                    se_diff = pooled_std * np.sqrt(1 / len(holiday_returns) + 1 / len(normal_returns))
                    df = len(holiday_returns) + len(normal_returns) - 2
                    t_critical = stats.t.ppf(0.975, df)

                    mean_diff = holiday_returns.mean() - normal_returns.mean()
                    ci_lower = mean_diff - t_critical * se_diff
                    ci_upper = mean_diff + t_critical * se_diff

                    results[factor] = MechanismResult(
                        factor_name=factor,
                        mechanism_type=MechanismType.CULTURAL,
                        effect_size=effect_size,
                        statistical_significance=p_value,
                        confidence_interval=(ci_lower, ci_upper),
                        affected_periods=self.market_data[holiday_mask].index.tolist(),
                        explanation=f"Holiday effect for {factor}: {mean_diff:.4f} mean difference",
                    )

        return results

    def analyze_earnings_season_effects(self) -> Dict[str, MechanismResult]:
        earnings_factors = [
            "earnings_season_q1",
            "earnings_season_q2",
            "earnings_season_q3",
            "earnings_season_q4",
        ]

        results = {}

        for factor in earnings_factors:
            if factor in self.market_data.columns:
                earnings_mask = self.market_data[factor] == 1

                if earnings_mask.sum() > 0:
                    earnings_returns = self.market_data.loc[earnings_mask, "returns"]
                    non_earnings_returns = self.market_data.loc[~earnings_mask, "returns"]

                    earnings_vol = earnings_returns.std()
                    non_earnings_vol = non_earnings_returns.std()

                    f_stat = earnings_vol**2 / non_earnings_vol**2
                    f_pvalue = 1 - stats.f.cdf(f_stat, len(earnings_returns) - 1, len(non_earnings_returns) - 1)

                    t_stat, t_pvalue = stats.ttest_ind(earnings_returns, non_earnings_returns)

                    results[factor] = MechanismResult(
                        factor_name=factor,
                        mechanism_type=MechanismType.ECONOMIC,
                        effect_size=f_stat,
                        statistical_significance=min(t_pvalue, f_pvalue),
                        confidence_interval=(
                            earnings_vol - non_earnings_vol,
                            earnings_vol + non_earnings_vol,
                        ),
                        affected_periods=self.market_data[earnings_mask].index.tolist(),
                        explanation=f"Earnings season volatility effect: {earnings_vol:.4f} vs {non_earnings_vol:.4f}",
                    )

        return results

    def analyze_institutional_flows(self) -> Dict[str, MechanismResult]:
        results = {}

        pension_mask = self.market_data["pension_rebalancing"] == 1
        if pension_mask.sum() > 0:
            pension_returns = self.market_data.loc[pension_mask, "returns"]
            normal_returns = self.market_data.loc[~pension_mask, "returns"]

            t_stat, p_value = stats.ttest_ind(pension_returns, normal_returns)

            effect_size = (pension_returns.mean() - normal_returns.mean()) / normal_returns.std()

            results["pension_rebalancing"] = MechanismResult(
                factor_name="pension_rebalancing",
                mechanism_type=MechanismType.INSTITUTIONAL,
                effect_size=effect_size,
                statistical_significance=p_value,
                confidence_interval=(
                    pension_returns.mean() - 1.96 * pension_returns.std() / np.sqrt(len(pension_returns)),
                    pension_returns.mean() + 1.96 * pension_returns.std() / np.sqrt(len(pension_returns)),
                ),
                affected_periods=self.market_data[pension_mask].index.tolist(),
                explanation=f"Pension rebalancing effect: {effect_size:.4f} standardized effect size",
            )

        for bonus_period in ["summer_bonus", "winter_bonus"]:
            if bonus_period in self.market_data.columns:
                bonus_mask = self.market_data[bonus_period] == 1

                if bonus_mask.sum() > 0:
                    bonus_returns = self.market_data.loc[bonus_mask, "returns"]
                    normal_returns = self.market_data.loc[~bonus_mask, "returns"]

                    t_stat, p_value = stats.ttest_ind(bonus_returns, normal_returns)
                    effect_size = (bonus_returns.mean() - normal_returns.mean()) / normal_returns.std()

                    results[bonus_period] = MechanismResult(
                        factor_name=bonus_period,
                        mechanism_type=MechanismType.ECONOMIC,
                        effect_size=effect_size,
                        statistical_significance=p_value,
                        confidence_interval=(
                            bonus_returns.mean() - 1.96 * bonus_returns.std() / np.sqrt(len(bonus_returns)),
                            bonus_returns.mean() + 1.96 * bonus_returns.std() / np.sqrt(len(bonus_returns)),
                        ),
                        affected_periods=self.market_data[bonus_mask].index.tolist(),
                        explanation=f"Bonus payment effect for {bonus_period}: {effect_size:.4f} standardized effect",
                    )

        return results

    def feature_importance_analysis(self) -> Dict[str, float]:
        mechanism_features = [
            "fiscal_year_end",
            "fiscal_year_start",
            "golden_week",
            "year_end_holidays",
            "obon_period",
            "earnings_season_q1",
            "earnings_season_q2",
            "earnings_season_q3",
            "earnings_season_q4",
            "summer_bonus",
            "winter_bonus",
            "month_end",
            "month_beginning",
            "pension_rebalancing",
            "tax_loss_selling",
        ]

        for month in range(1, 13):
            col_name = f"month_{month}"
            self.market_data[col_name] = (self.market_data.index.month == month).astype(int)
            mechanism_features.append(col_name)

        feature_data = self.market_data[mechanism_features + ["returns"]].dropna()
        X = feature_data[mechanism_features]
        y = feature_data["returns"]

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importance_dict = dict(zip(mechanism_features, rf.feature_importances_))

        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        self.logger.info(f"Feature importance analysis complete. Top factor: {list(sorted_importance.keys())[0]}")

        return sorted_importance

    def comprehensive_mechanism_analysis(self) -> Dict[str, Any]:
        results = {
            "fiscal_year_effects": self.analyze_fiscal_year_effects(),
            "holiday_effects": self.analyze_holiday_effects(),
            "earnings_season_effects": self.analyze_earnings_season_effects(),
            "institutional_flow_effects": self.analyze_institutional_flows(),
            "feature_importance": self.feature_importance_analysis(),
        }

        significant_mechanisms = []

        for category, mechanism_results in results.items():
            if category == "feature_importance":
                continue

            if isinstance(mechanism_results, dict):
                for mechanism_name, result in mechanism_results.items():
                    if result.statistical_significance < 0.05:
                        significant_mechanisms.append(
                            {
                                "category": category,
                                "mechanism": mechanism_name,
                                "p_value": result.statistical_significance,
                                "effect_size": result.effect_size,
                            }
                        )
            else:
                if mechanism_results.statistical_significance < 0.05:
                    significant_mechanisms.append(
                        {
                            "category": category,
                            "mechanism": mechanism_results.factor_name,
                            "p_value": mechanism_results.statistical_significance,
                            "effect_size": mechanism_results.effect_size,
                        }
                    )

        significant_mechanisms.sort(key=lambda x: x["p_value"])

        results["summary"] = {
            "total_mechanisms_tested": sum(
                len(v) if isinstance(v, dict) else 1 for k, v in results.items() if k != "feature_importance"
            ),
            "significant_mechanisms": significant_mechanisms,
            "most_significant": significant_mechanisms[0] if significant_mechanisms else None,
            "top_features_by_importance": list(results["feature_importance"].items())[:5],
        }

        return results

    def validate_mechanisms_out_of_sample(self, train_end_date: datetime, test_start_date: datetime) -> Dict[str, Any]:
        train_data = self.market_data[self.market_data.index <= train_end_date]
        test_data = self.market_data[self.market_data.index >= test_start_date]

        if len(train_data) == 0 or len(test_data) == 0:
            return {"error": "Insufficient data for train/test split"}

        train_analyzer = MechanismAnalyzer(train_data)
        train_results = train_analyzer.comprehensive_mechanism_analysis()

        test_analyzer = MechanismAnalyzer(test_data)
        test_results = test_analyzer.comprehensive_mechanism_analysis()

        validation_results = {
            "train_period": {
                "start": train_data.index.min(),
                "end": train_data.index.max(),
                "observations": len(train_data),
            },
            "test_period": {
                "start": test_data.index.min(),
                "end": test_data.index.max(),
                "observations": len(test_data),
            },
            "consistency_analysis": self._compare_mechanism_results(train_results, test_results),
        }

        return validation_results

    def _compare_mechanism_results(self, train_results: Dict[str, Any], test_results: Dict[str, Any]) -> Dict[str, Any]:
        consistent_mechanisms = []

        train_significant = {m["mechanism"] for m in train_results["summary"]["significant_mechanisms"]}
        test_significant = {m["mechanism"] for m in test_results["summary"]["significant_mechanisms"]}

        consistent_mechanisms = list(train_significant.intersection(test_significant))
        only_in_train = list(train_significant - test_significant)
        only_in_test = list(test_significant - train_significant)

        return {
            "consistent_mechanisms": consistent_mechanisms,
            "only_significant_in_train": only_in_train,
            "only_significant_in_test": only_in_test,
            "consistency_rate": len(consistent_mechanisms) / max(len(train_significant), 1),
            "total_mechanisms_tested": len(train_significant.union(test_significant)),
        }
