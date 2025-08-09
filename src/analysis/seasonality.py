"""
Seasonality analysis module for detecting and quantifying seasonal patterns.

This module implements comprehensive statistical testing for seasonal patterns
including t-tests, ANOVA, regression analysis, and multiple comparison corrections.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass

import scipy.stats as stats
from scipy.stats import jarque_bera, ttest_1samp, f_oneway
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from ..config import get_logger, JapaneseMarketConstants


@dataclass
class SeasonalityResult:
    """Result of seasonality analysis."""
    
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
    """Core statistical analysis for seasonal patterns."""
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 significance_level: float = 0.05,
                 min_observations: int = 20):
        """
        Initialize the seasonality analyzer.
        
        Args:
            data: DataFrame with datetime index and returns column
            significance_level: Statistical significance threshold
            min_observations: Minimum observations required per period
        """
        self.data = data.copy()
        self.alpha = significance_level
        self.min_observations = min_observations
        self.logger = get_logger(__name__)
        
        # Prepare data
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data for analysis."""
        # Ensure we have returns column
        if 'returns' not in self.data.columns:
            if 'adjusted_close' in self.data.columns:
                self.data['returns'] = self.data['adjusted_close'].pct_change()
            elif 'close_price' in self.data.columns:
                self.data['returns'] = self.data['close_price'].pct_change()
            else:
                raise ValueError("No price or returns data found")
        
        # Add time-based features
        self.data['month'] = self.data.index.month
        self.data['year'] = self.data.index.year
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['quarter'] = self.data.index.quarter
        
        # Add Japanese-specific features
        self.data['fiscal_year_end'] = (
            (self.data.index.month == JapaneseMarketConstants.FISCAL_YEAR_END_MONTH) & 
            (self.data.index.day >= 20)
        ).astype(int)
        
        self.data['golden_week'] = (
            (self.data.index.month == JapaneseMarketConstants.GOLDEN_WEEK_START[0]) & 
            (self.data.index.day >= JapaneseMarketConstants.GOLDEN_WEEK_START[1]) &
            (self.data.index.day <= JapaneseMarketConstants.GOLDEN_WEEK_END[1])
        ).astype(int)
        
        # Remove NaN values
        self.data = self.data.dropna(subset=['returns'])
        
        self.logger.info(f"Prepared data: {len(self.data)} observations from {self.data.index.min()} to {self.data.index.max()}")
    
    def test_monthly_patterns(self, 
                             multiple_testing_correction: str = 'bonferroni') -> Dict[int, SeasonalityResult]:
        """
        Comprehensive monthly seasonality analysis.
        
        Args:
            multiple_testing_correction: Method for multiple testing correction
            
        Returns:
            Dictionary of seasonality results by month
        """
        self.logger.info("Starting monthly seasonality analysis")
        
        results = {}
        monthly_returns = {}
        
        # Group returns by month
        for month in range(1, 13):
            month_data = self.data[self.data['month'] == month]['returns']
            if len(month_data) >= self.min_observations:
                monthly_returns[month] = month_data
            else:
                self.logger.warning(f"Insufficient observations for month {month}: {len(month_data)}")
        
        # Calculate statistics for each month
        for month, returns in monthly_returns.items():
            # Descriptive statistics
            mean_return = returns.mean()
            std_return = returns.std()
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
            
            # t-test against zero
            t_stat, t_pvalue = ttest_1samp(returns, 0.0)
            
            # Normality test
            jb_stat, jb_pvalue = jarque_bera(returns)
            is_normal = jb_pvalue > self.alpha
            
            # Confidence interval
            confidence_interval = stats.t.interval(
                1 - self.alpha,
                len(returns) - 1,
                loc=mean_return,
                scale=stats.sem(returns)
            )
            
            results[month] = SeasonalityResult(
                month=month,
                mean_return=mean_return,
                std_return=std_return,
                skewness=skew,
                kurtosis=kurt,
                t_statistic=t_stat,
                t_pvalue=t_pvalue,
                is_significant=False,  # Will be updated after correction
                normality_test_statistic=jb_stat,
                normality_pvalue=jb_pvalue,
                is_normal=is_normal,
                sample_size=len(returns),
                confidence_interval=confidence_interval
            )
        
        # Apply multiple testing correction
        if len(results) > 1:
            p_values = [result.t_pvalue for result in results.values()]
            corrected_pvalues = multipletests(p_values, method=multiple_testing_correction)[1]
            
            for i, month in enumerate(results.keys()):
                results[month].is_significant = corrected_pvalues[i] < self.alpha
        
        self.logger.info(f"Monthly analysis complete. Found {sum(1 for r in results.values() if r.is_significant)} significant months")
        
        return results
    
    def test_day_of_week_patterns(self) -> Dict[int, Dict[str, Any]]:
        """Analyze day-of-week effects."""
        self.logger.info("Analyzing day-of-week patterns")
        
        results = {}
        
        for dow in range(5):  # Monday=0 to Friday=4
            dow_data = self.data[self.data['day_of_week'] == dow]['returns']
            
            if len(dow_data) >= self.min_observations:
                # Basic statistics
                mean_return = dow_data.mean()
                std_return = dow_data.std()
                
                # t-test against zero
                t_stat, p_value = ttest_1samp(dow_data, 0.0)
                
                results[dow] = {
                    'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][dow],
                    'mean_return': mean_return,
                    'std_return': std_return,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'is_significant': p_value < self.alpha,
                    'sample_size': len(dow_data)
                }
        
        return results
    
    def test_quarter_patterns(self) -> Dict[int, Dict[str, Any]]:
        """Analyze quarterly patterns."""
        self.logger.info("Analyzing quarterly patterns")
        
        results = {}
        
        for quarter in range(1, 5):
            quarter_data = self.data[self.data['quarter'] == quarter]['returns']
            
            if len(quarter_data) >= self.min_observations:
                mean_return = quarter_data.mean()
                std_return = quarter_data.std()
                
                # t-test against zero
                t_stat, p_value = ttest_1samp(quarter_data, 0.0)
                
                results[quarter] = {
                    'quarter': quarter,
                    'mean_return': mean_return,
                    'std_return': std_return,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'is_significant': p_value < self.alpha,
                    'sample_size': len(quarter_data)
                }
        
        return results
    
    def rolling_seasonality_analysis(self, 
                                   window_years: int = 5,
                                   step_months: int = 6) -> Dict[str, Any]:
        """
        Perform rolling window analysis to test stability of seasonal patterns.
        
        Args:
            window_years: Size of rolling window in years
            step_months: Step size in months
            
        Returns:
            Rolling analysis results
        """
        self.logger.info(f"Starting rolling seasonality analysis with {window_years}-year windows")
        
        window_days = window_years * 252  # Approximate trading days
        step_days = step_months * 21     # Approximate trading days per month
        
        rolling_results = []
        
        start_idx = 0
        while start_idx + window_days <= len(self.data):
            end_idx = start_idx + window_days
            
            # Extract window data
            window_data = self.data.iloc[start_idx:end_idx]
            
            # Create temporary analyzer for this window
            window_analyzer = SeasonalityAnalyzer(
                window_data,
                significance_level=self.alpha,
                min_observations=self.min_observations
            )
            
            # Analyze this window
            monthly_results = window_analyzer.test_monthly_patterns()
            
            # Store results
            window_result = {
                'window_start': window_data.index.min(),
                'window_end': window_data.index.max(),
                'significant_months': [
                    month for month, result in monthly_results.items()
                    if result.is_significant
                ],
                'monthly_means': {
                    month: result.mean_return 
                    for month, result in monthly_results.items()
                }
            }
            
            rolling_results.append(window_result)
            start_idx += step_days
        
        # Analyze consistency across windows
        consistency_analysis = self._analyze_pattern_consistency(rolling_results)
        
        return {
            'rolling_windows': rolling_results,
            'consistency_analysis': consistency_analysis
        }
    
    def _analyze_pattern_consistency(self, rolling_results: List[Dict]) -> Dict[str, Any]:
        """Analyze consistency of patterns across rolling windows."""
        
        # Count how often each month is significant
        month_significance_count = {month: 0 for month in range(1, 13)}
        
        for window in rolling_results:
            for month in window['significant_months']:
                month_significance_count[month] += 1
        
        total_windows = len(rolling_results)
        consistency_threshold = 0.6  # 60% of windows
        
        consistent_months = {
            month: count / total_windows 
            for month, count in month_significance_count.items()
            if count / total_windows >= consistency_threshold
        }
        
        return {
            'total_windows': total_windows,
            'month_significance_frequency': {
                month: count / total_windows 
                for month, count in month_significance_count.items()
            },
            'consistent_months': consistent_months,
            'consistency_threshold': consistency_threshold
        }
    
    def compare_periods(self, 
                       period1_start: datetime, 
                       period1_end: datetime,
                       period2_start: datetime, 
                       period2_end: datetime) -> Dict[str, Any]:
        """Compare seasonal patterns between two periods."""
        
        period1_data = self.data[
            (self.data.index >= period1_start) & 
            (self.data.index <= period1_end)
        ]
        
        period2_data = self.data[
            (self.data.index >= period2_start) & 
            (self.data.index <= period2_end)
        ]
        
        comparison_results = {}
        
        for month in range(1, 13):
            p1_returns = period1_data[period1_data['month'] == month]['returns']
            p2_returns = period2_data[period2_data['month'] == month]['returns']
            
            if len(p1_returns) >= self.min_observations and len(p2_returns) >= self.min_observations:
                # Two-sample t-test
                t_stat, p_value = stats.ttest_ind(p1_returns, p2_returns)
                
                comparison_results[month] = {
                    'period1_mean': p1_returns.mean(),
                    'period2_mean': p2_returns.mean(),
                    'difference': p2_returns.mean() - p1_returns.mean(),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'is_significantly_different': p_value < self.alpha,
                    'period1_sample_size': len(p1_returns),
                    'period2_sample_size': len(p2_returns)
                }
        
        return comparison_results
    
    def seasonal_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive seasonality summary report."""
        
        # Monthly analysis
        monthly_results = self.test_monthly_patterns()
        
        # Day of week analysis
        dow_results = self.test_day_of_week_patterns()
        
        # Quarter analysis
        quarter_results = self.test_quarter_patterns()
        
        # Overall statistics
        overall_stats = {
            'total_observations': len(self.data),
            'date_range': {
                'start': self.data.index.min(),
                'end': self.data.index.max()
            },
            'overall_mean_return': self.data['returns'].mean(),
            'overall_volatility': self.data['returns'].std(),
            'skewness': stats.skew(self.data['returns']),
            'kurtosis': stats.kurtosis(self.data['returns'])
        }
        
        # Significant patterns summary
        significant_months = [
            month for month, result in monthly_results.items() 
            if result.is_significant
        ]
        
        significant_dows = [
            result['day_name'] for dow, result in dow_results.items()
            if result['is_significant']
        ]
        
        return {
            'overall_statistics': overall_stats,
            'monthly_results': monthly_results,
            'day_of_week_results': dow_results,
            'quarterly_results': quarter_results,
            'summary': {
                'significant_months': significant_months,
                'significant_days_of_week': significant_dows,
                'strongest_month': max(
                    monthly_results.items(), 
                    key=lambda x: abs(x[1].mean_return)
                )[0] if monthly_results else None,
                'weakest_month': min(
                    monthly_results.items(), 
                    key=lambda x: x[1].mean_return
                )[0] if monthly_results else None
            }
        }


class SeasonalRegressionModel:
    """Advanced regression models for seasonal analysis."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize regression model with prepared data."""
        self.data = data.copy()
        self.logger = get_logger(__name__)
        self._prepare_features()
    
    def _prepare_features(self):
        """Create seasonal and control variables."""
        # Seasonal dummies (month 1 is reference)
        for month in range(2, 13):
            self.data[f'month_{month}'] = (self.data.index.month == month).astype(int)
        
        # Day of week dummies (Monday is reference)
        for dow in range(1, 5):  # Tuesday through Friday
            dow_names = ['tuesday', 'wednesday', 'thursday', 'friday']
            self.data[f'dow_{dow_names[dow-1]}'] = (self.data.index.dayofweek == dow).astype(int)
        
        # Quarter dummies (Q1 is reference)
        for quarter in range(2, 5):
            self.data[f'quarter_{quarter}'] = (self.data.index.quarter == quarter).astype(int)
        
        # Control variables
        self.data['year'] = self.data.index.year
        self.data['year_normalized'] = (self.data['year'] - self.data['year'].min()) / (self.data['year'].max() - self.data['year'].min())
        
        # Japanese-specific variables
        self.data['fiscal_year_end'] = (
            (self.data.index.month == JapaneseMarketConstants.FISCAL_YEAR_END_MONTH) & 
            (self.data.index.day >= 20)
        ).astype(int)
        
        self.data['golden_week'] = (
            (self.data.index.month == JapaneseMarketConstants.GOLDEN_WEEK_START[0]) & 
            (self.data.index.day >= JapaneseMarketConstants.GOLDEN_WEEK_START[1]) &
            (self.data.index.day <= JapaneseMarketConstants.GOLDEN_WEEK_END[1])
        ).astype(int)
        
        # Lagged returns for momentum/reversal effects
        self.data['returns_lag1'] = self.data['returns'].shift(1)
        self.data['returns_lag5'] = self.data['returns'].shift(5)
        
        self.logger.info("Features prepared for regression analysis")
    
    def fit_seasonal_model(self, 
                          include_controls: bool = True,
                          robust_se: bool = True) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Fit comprehensive seasonal regression model.
        
        Args:
            include_controls: Whether to include control variables
            robust_se: Whether to use robust standard errors
            
        Returns:
            Fitted regression model
        """
        # Define model specification
        month_vars = [f'month_{i}' for i in range(2, 13)]
        
        features = month_vars.copy()
        
        if include_controls:
            dow_vars = ['dow_tuesday', 'dow_wednesday', 'dow_thursday', 'dow_friday']
            quarter_vars = [f'quarter_{i}' for i in range(2, 5)]
            japanese_vars = ['fiscal_year_end', 'golden_week']
            lag_vars = ['returns_lag1', 'returns_lag5']
            
            features.extend(dow_vars)
            features.extend(quarter_vars) 
            features.extend(japanese_vars)
            features.extend(lag_vars)
            features.append('year_normalized')
        
        # Prepare data (remove NaN values)
        model_data = self.data[['returns'] + features].dropna()
        
        # Fit model
        X = model_data[features]
        X = sm.add_constant(X)  # Add intercept
        y = model_data['returns']
        
        if robust_se:
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        else:
            model = sm.OLS(y, X).fit()
        
        self.logger.info(f"Fitted seasonal regression model with {len(features)} features")
        
        return model
    
    def test_joint_significance(self, model, feature_groups: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """Test joint significance of feature groups."""
        results = {}
        
        for group_name, features in feature_groups.items():
            # Get parameter names that match features
            param_names = [f for f in features if f in model.params.index]
            
            if param_names:
                # F-test for joint significance
                f_test = model.f_test([f"{param} = 0" for param in param_names])
                
                results[group_name] = {
                    'f_statistic': f_test.fvalue,
                    'p_value': f_test.pvalue,
                    'is_significant': f_test.pvalue < 0.05,
                    'features_tested': param_names
                }
        
        return results
    
    def predict_seasonal_returns(self, 
                                model, 
                                future_dates: pd.DatetimeIndex) -> pd.Series:
        """Predict returns for future dates using fitted model."""
        
        # Create feature matrix for future dates
        future_df = pd.DataFrame(index=future_dates)
        
        # Monthly dummies
        for month in range(2, 13):
            future_df[f'month_{month}'] = (future_df.index.month == month).astype(int)
        
        # Day of week dummies
        dow_names = ['tuesday', 'wednesday', 'thursday', 'friday']
        for dow in range(1, 5):
            future_df[f'dow_{dow_names[dow-1]}'] = (future_df.index.dayofweek == dow).astype(int)
        
        # Other features (set to median values or zero for simplicity)
        for feature in model.params.index:
            if feature not in future_df.columns and feature != 'const':
                future_df[feature] = 0
        
        # Add constant
        future_df = sm.add_constant(future_df)
        
        # Ensure column order matches model
        feature_cols = [col for col in model.params.index if col in future_df.columns]
        X_future = future_df[feature_cols]
        
        # Predict
        predictions = model.predict(X_future)
        
        return predictions