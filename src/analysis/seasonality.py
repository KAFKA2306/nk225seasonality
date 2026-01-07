from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import jarque_bera, ttest_1samp
from statsmodels.stats.multitest import multipletests

from ..config import JapaneseMarketConstants


@dataclass
class SeasonalityResult:
    month: int
    mean_return: float
    std_return: float
    skewness: float
    kurtosis: float
    t_statistic: float
    t_pvalue: float
    is_significant: bool
    normality_test_statistic: float
    normality_pvalue: bool
    is_normal: bool
    sample_size: int
    confidence_interval: Tuple[float, float]


class SeasonalityAnalyzer:
    def __init__(
        self,
        data: pd.DataFrame,
        significance_level: float = 0.05,
        min_observations: int = 20,
    ):
        self.data = data.copy()
        self.alpha = significance_level
        self.min_observations = min_observations
        self._prepare_data()

    def _prepare_data(self):
        if "returns" not in self.data.columns:
            self.data["returns"] = self.data["close_price"].pct_change()

        self.data["month"] = self.data.index.month
        self.data["year"] = self.data.index.year
        self.data["day_of_week"] = self.data.index.dayofweek
        self.data["quarter"] = self.data.index.quarter

        self.data["fiscal_year_end"] = (
            (self.data.index.month == JapaneseMarketConstants.FISCAL_YEAR_END_MONTH) & (self.data.index.day >= 20)
        ).astype(int)

        self.data["golden_week"] = (
            (self.data.index.month == JapaneseMarketConstants.GOLDEN_WEEK_START[0])
            & (self.data.index.day >= JapaneseMarketConstants.GOLDEN_WEEK_START[1])
            & (self.data.index.day <= JapaneseMarketConstants.GOLDEN_WEEK_END[1])
        ).astype(int)

        self.data = self.data.dropna(subset=["returns"])

    def test_monthly_patterns(self, multiple_testing_correction: str = "bonferroni") -> Dict[int, SeasonalityResult]:
        results = {}
        monthly_returns = {}

        for month in range(1, 13):
            month_data = self.data[self.data["month"] == month]["returns"]
            if len(month_data) >= self.min_observations:
                monthly_returns[month] = month_data

        for month, returns in monthly_returns.items():
            mean_return = returns.mean()
            std_return = returns.std()
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
            t_stat, t_pvalue = ttest_1samp(returns, 0.0)
            jb_stat, jb_pvalue = jarque_bera(returns)
            is_normal = jb_pvalue > self.alpha

            confidence_interval = stats.t.interval(
                1 - self.alpha,
                len(returns) - 1,
                loc=mean_return,
                scale=stats.sem(returns),
            )

            results[month] = SeasonalityResult(
                month=month,
                mean_return=mean_return,
                std_return=std_return,
                skewness=skew,
                kurtosis=kurt,
                t_statistic=t_stat,
                t_pvalue=t_pvalue,
                is_significant=False,
                normality_test_statistic=jb_stat,
                normality_pvalue=jb_pvalue,
                is_normal=is_normal,
                sample_size=len(returns),
                confidence_interval=confidence_interval,
            )

        if len(results) > 1:
            p_values = [result.t_pvalue for result in results.values()]
            corrected_pvalues = multipletests(p_values, method=multiple_testing_correction)[1]

            for i, month in enumerate(results.keys()):
                results[month].is_significant = corrected_pvalues[i] < self.alpha

        return results

    def test_day_of_week_patterns(self) -> Dict[int, Dict[str, Any]]:
        results = {}
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        for dow in range(5):
            dow_data = self.data[self.data["day_of_week"] == dow]["returns"]
            if len(dow_data) >= self.min_observations:
                mean_return = dow_data.mean()
                std_return = dow_data.std()
                t_stat, p_value = ttest_1samp(dow_data, 0.0)

                results[dow] = {
                    "day_name": days[dow],
                    "mean_return": mean_return,
                    "std_return": std_return,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "is_significant": p_value < self.alpha,
                    "sample_size": len(dow_data),
                }
        return results

    def test_quarter_patterns(self) -> Dict[int, Dict[str, Any]]:
        results = {}
        for quarter in range(1, 5):
            quarter_data = self.data[self.data["quarter"] == quarter]["returns"]
            if len(quarter_data) >= self.min_observations:
                mean_return = quarter_data.mean()
                std_return = quarter_data.std()
                t_stat, p_value = ttest_1samp(quarter_data, 0.0)
                results[quarter] = {
                    "quarter": quarter,
                    "mean_return": mean_return,
                    "std_return": std_return,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "is_significant": p_value < self.alpha,
                    "sample_size": len(quarter_data),
                }
        return results

    def rolling_seasonality_analysis(self, window_years: int = 5, step_months: int = 6) -> Dict[str, Any]:
        window_days = window_years * 252
        step_days = step_months * 21
        rolling_results = []
        start_idx = 0

        while start_idx + window_days <= len(self.data):
            end_idx = start_idx + window_days
            window_data = self.data.iloc[start_idx:end_idx]
            window_analyzer = SeasonalityAnalyzer(window_data, self.alpha, self.min_observations)
            monthly_results = window_analyzer.test_monthly_patterns()

            window_result = {
                "window_start": window_data.index.min(),
                "window_end": window_data.index.max(),
                "significant_months": [m for m, r in monthly_results.items() if r.is_significant],
                "monthly_means": {m: r.mean_return for m, r in monthly_results.items()},
            }
            rolling_results.append(window_result)
            start_idx += step_days

        consistency_analysis = self._analyze_pattern_consistency(rolling_results)
        return {
            "rolling_windows": rolling_results,
            "consistency_analysis": consistency_analysis,
        }

    def _analyze_pattern_consistency(self, rolling_results: List[Dict]) -> Dict[str, Any]:
        month_significance_count = {month: 0 for month in range(1, 13)}

        for window in rolling_results:
            for month in window["significant_months"]:
                month_significance_count[month] += 1

        total_windows = len(rolling_results)
        consistency_threshold = 0.6

        consistent_months = {
            month: count / total_windows
            for month, count in month_significance_count.items()
            if count / total_windows >= consistency_threshold
        }

        return {
            "total_windows": total_windows,
            "month_significance_frequency": {m: c / total_windows for m, c in month_significance_count.items()},
            "consistent_months": consistent_months,
            "consistency_threshold": consistency_threshold,
        }

    def seasonal_summary_report(self) -> Dict[str, Any]:
        monthly_results = self.test_monthly_patterns()
        dow_results = self.test_day_of_week_patterns()
        quarter_results = self.test_quarter_patterns()

        overall_stats = {
            "total_observations": len(self.data),
            "date_range": {
                "start": self.data.index.min(),
                "end": self.data.index.max(),
            },
            "overall_mean_return": self.data["returns"].mean(),
            "overall_volatility": self.data["returns"].std(),
            "skewness": stats.skew(self.data["returns"]),
            "kurtosis": stats.kurtosis(self.data["returns"]),
        }

        significant_months = [m for m, r in monthly_results.items() if r.is_significant]
        significant_dows = [r["day_name"] for dow, r in dow_results.items() if r["is_significant"]]

        strongest_month = (
            max(monthly_results.items(), key=lambda x: abs(x[1].mean_return))[0] if monthly_results else None
        )
        weakest_month = min(monthly_results.items(), key=lambda x: x[1].mean_return)[0] if monthly_results else None

        return {
            "overall_statistics": overall_stats,
            "monthly_results": monthly_results,
            "day_of_week_results": dow_results,
            "quarterly_results": quarter_results,
            "summary": {
                "significant_months": significant_months,
                "significant_days_of_week": significant_dows,
                "strongest_month": strongest_month,
                "weakest_month": weakest_month,
            },
        }


class SeasonalRegressionModel:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self._prepare_features()

    def _prepare_features(self):
        for month in range(2, 13):
            self.data[f"month_{month}"] = (self.data.index.month == month).astype(int)

        dow_names = ["tuesday", "wednesday", "thursday", "friday"]
        for dow in range(1, 5):
            self.data[f"dow_{dow_names[dow - 1]}"] = (self.data.index.dayofweek == dow).astype(int)

        for quarter in range(2, 5):
            self.data[f"quarter_{quarter}"] = (self.data.index.quarter == quarter).astype(int)

        self.data["year"] = self.data.index.year
        self.data["year_normalized"] = (self.data["year"] - self.data["year"].min()) / (
            self.data["year"].max() - self.data["year"].min()
        )

        self.data["fiscal_year_end"] = (
            (self.data.index.month == JapaneseMarketConstants.FISCAL_YEAR_END_MONTH) & (self.data.index.day >= 20)
        ).astype(int)

        self.data["golden_week"] = (
            (self.data.index.month == JapaneseMarketConstants.GOLDEN_WEEK_START[0])
            & (self.data.index.day >= JapaneseMarketConstants.GOLDEN_WEEK_START[1])
            & (self.data.index.day <= JapaneseMarketConstants.GOLDEN_WEEK_END[1])
        ).astype(int)

        self.data["returns_lag1"] = self.data["returns"].shift(1)
        self.data["returns_lag5"] = self.data["returns"].shift(5)

    def fit_seasonal_model(self, include_controls: bool = True, robust_se: bool = True):
        month_vars = [f"month_{i}" for i in range(2, 13)]
        features = month_vars.copy()

        if include_controls:
            features.extend(["dow_tuesday", "dow_wednesday", "dow_thursday", "dow_friday"])
            features.extend([f"quarter_{i}" for i in range(2, 5)])
            features.extend(["fiscal_year_end", "golden_week"])
            features.extend(["returns_lag1", "returns_lag5"])
            features.append("year_normalized")

        model_data = self.data[["returns"] + features].dropna()
        X = sm.add_constant(model_data[features])
        y = model_data["returns"]

        if robust_se:
            return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
        return sm.OLS(y, X).fit()
