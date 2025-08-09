"""
Value-at-Risk (VaR) and Expected Shortfall calculation module.

This module provides various methods for calculating VaR and Expected Shortfall
including parametric, historical, and Monte Carlo approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from scipy.stats import norm, t
from sklearn.mixture import GaussianMixture

from ..config import get_logger


class VaRMethod(Enum):
    """Methods for VaR calculation."""
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"
    EXTREME_VALUE = "extreme_value"


@dataclass
class VaRResult:
    """Result of VaR calculation."""
    
    confidence_level: float
    var_value: float
    expected_shortfall: float
    method: str
    sample_size: int
    volatility: float
    mean_return: float


class VaRCalculator:
    """Value-at-Risk calculator with multiple methodologies."""
    
    def __init__(self):
        """Initialize VaR calculator."""
        self.logger = get_logger(__name__)
    
    def calculate_var(self, 
                     returns: np.ndarray,
                     confidence_level: float = 0.95,
                     method: VaRMethod = VaRMethod.HISTORICAL,
                     holding_period: int = 1) -> VaRResult:
        """
        Calculate Value-at-Risk using specified method.
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            method: Method to use for VaR calculation
            holding_period: Holding period in days
            
        Returns:
            VaR result
        """
        
        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")
        
        # Scale returns for holding period
        if holding_period > 1:
            scaled_returns = returns * np.sqrt(holding_period)
        else:
            scaled_returns = returns
        
        if method == VaRMethod.PARAMETRIC:
            return self._parametric_var(scaled_returns, confidence_level)
        
        elif method == VaRMethod.HISTORICAL:
            return self._historical_var(scaled_returns, confidence_level)
        
        elif method == VaRMethod.MONTE_CARLO:
            return self._monte_carlo_var(scaled_returns, confidence_level)
        
        elif method == VaRMethod.CORNISH_FISHER:
            return self._cornish_fisher_var(scaled_returns, confidence_level)
        
        elif method == VaRMethod.EXTREME_VALUE:
            return self._extreme_value_var(scaled_returns, confidence_level)
        
        else:
            raise ValueError(f"Unsupported VaR method: {method}")
    
    def _parametric_var(self, returns: np.ndarray, confidence_level: float) -> VaRResult:
        """Calculate parametric VaR assuming normal distribution."""
        
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Calculate VaR using normal distribution quantile
        alpha = 1 - confidence_level
        z_score = norm.ppf(alpha)
        var_value = -(mean_return + z_score * volatility)
        
        # Expected Shortfall (Conditional VaR)
        expected_shortfall = -(mean_return + volatility * norm.pdf(z_score) / alpha)
        
        return VaRResult(
            confidence_level=confidence_level,
            var_value=var_value,
            expected_shortfall=expected_shortfall,
            method="parametric_normal",
            sample_size=len(returns),
            volatility=volatility,
            mean_return=mean_return
        )
    
    def _historical_var(self, returns: np.ndarray, confidence_level: float) -> VaRResult:
        """Calculate historical VaR using empirical distribution."""
        
        # Sort returns in ascending order (worst to best)
        sorted_returns = np.sort(returns)
        
        # Calculate VaR as percentile
        alpha = 1 - confidence_level
        var_percentile = alpha * 100
        var_value = -np.percentile(returns, var_percentile)
        
        # Calculate Expected Shortfall as mean of tail
        tail_index = int(np.ceil(alpha * len(returns)))
        if tail_index == 0:
            tail_index = 1
        
        tail_returns = sorted_returns[:tail_index]
        expected_shortfall = -np.mean(tail_returns)
        
        return VaRResult(
            confidence_level=confidence_level,
            var_value=var_value,
            expected_shortfall=expected_shortfall,
            method="historical",
            sample_size=len(returns),
            volatility=np.std(returns),
            mean_return=np.mean(returns)
        )
    
    def _monte_carlo_var(self, returns: np.ndarray, confidence_level: float,
                        num_simulations: int = 10000) -> VaRResult:
        """Calculate Monte Carlo VaR using bootstrapping."""
        
        # Bootstrap simulation
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.choice(returns, size=num_simulations, replace=True)
        
        # Calculate VaR from simulated distribution
        alpha = 1 - confidence_level
        var_value = -np.percentile(simulated_returns, alpha * 100)
        
        # Expected Shortfall
        tail_returns = simulated_returns[simulated_returns <= -var_value]
        expected_shortfall = -np.mean(tail_returns)
        
        return VaRResult(
            confidence_level=confidence_level,
            var_value=var_value,
            expected_shortfall=expected_shortfall,
            method="monte_carlo_bootstrap",
            sample_size=len(returns),
            volatility=np.std(returns),
            mean_return=np.mean(returns)
        )
    
    def _cornish_fisher_var(self, returns: np.ndarray, confidence_level: float) -> VaRResult:
        """Calculate VaR using Cornish-Fisher expansion for non-normal distributions."""
        
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Cornish-Fisher quantile adjustment
        alpha = 1 - confidence_level
        z = norm.ppf(alpha)
        
        # Cornish-Fisher expansion
        cf_quantile = (z + 
                      (z**2 - 1) * skewness / 6 +
                      (z**3 - 3*z) * kurtosis / 24 -
                      (2*z**3 - 5*z) * skewness**2 / 36)
        
        var_value = -(mean_return + cf_quantile * volatility)
        
        # Approximate Expected Shortfall
        # Using modified formula that accounts for skewness and kurtosis
        phi_z = norm.pdf(z)
        expected_shortfall = -(mean_return + volatility * phi_z / alpha *
                             (1 + skewness * (z**2 - 1) / 6 + 
                              kurtosis * z * (z**2 - 3) / 24))
        
        return VaRResult(
            confidence_level=confidence_level,
            var_value=var_value,
            expected_shortfall=expected_shortfall,
            method="cornish_fisher",
            sample_size=len(returns),
            volatility=volatility,
            mean_return=mean_return
        )
    
    def _extreme_value_var(self, returns: np.ndarray, confidence_level: float) -> VaRResult:
        """Calculate VaR using Extreme Value Theory (EVT)."""
        
        # Use Peaks Over Threshold (POT) method
        threshold_percentile = 90  # Use top 10% of losses as extreme values
        threshold = np.percentile(-returns, threshold_percentile)  # Convert to losses
        
        losses = -returns  # Convert to losses (positive values)
        excesses = losses[losses > threshold] - threshold
        
        if len(excesses) < 10:  # Need sufficient extreme observations
            self.logger.warning("Insufficient extreme observations for EVT, falling back to historical VaR")
            return self._historical_var(returns, confidence_level)
        
        try:
            # Fit Generalized Pareto Distribution (GPD) to excesses
            from scipy.stats import genpareto
            
            # Method of moments estimation for GPD parameters
            shape, loc, scale = genpareto.fit(excesses, floc=0)  # loc=0 for POT
            
            # Calculate VaR using EVT
            n = len(returns)
            nu = len(excesses)  # Number of excesses
            
            alpha = 1 - confidence_level
            
            if shape != 0:
                var_value = threshold + (scale / shape) * (
                    ((n / nu) * alpha)**(-shape) - 1
                )
            else:
                var_value = threshold + scale * np.log((n / nu) * alpha)
            
            # Expected Shortfall for GPD
            if shape < 1 and shape != 0:
                expected_shortfall = (var_value + scale - shape * threshold) / (1 - shape)
            else:
                # Fallback to historical ES
                tail_returns = returns[returns <= -var_value]
                expected_shortfall = -np.mean(tail_returns) if len(tail_returns) > 0 else var_value
            
            return VaRResult(
                confidence_level=confidence_level,
                var_value=var_value,
                expected_shortfall=expected_shortfall,
                method="extreme_value_theory",
                sample_size=len(returns),
                volatility=np.std(returns),
                mean_return=np.mean(returns)
            )
            
        except Exception as e:
            self.logger.error(f"EVT calculation failed: {e}, falling back to historical VaR")
            return self._historical_var(returns, confidence_level)
    
    def rolling_var(self, 
                   returns: pd.Series,
                   window: int = 252,
                   confidence_level: float = 0.95,
                   method: VaRMethod = VaRMethod.HISTORICAL) -> pd.Series:
        """
        Calculate rolling VaR over time.
        
        Args:
            returns: Time series of returns
            window: Rolling window size
            confidence_level: Confidence level
            method: VaR calculation method
            
        Returns:
            Time series of rolling VaR values
        """
        
        var_values = []
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i].values
            var_result = self.calculate_var(window_returns, confidence_level, method)
            var_values.append(var_result.var_value)
        
        # Create time series with appropriate index
        var_index = returns.index[window:]
        return pd.Series(var_values, index=var_index, name=f'VaR_{confidence_level}')
    
    def backtesting_kupiec_test(self, 
                               returns: np.ndarray,
                               var_forecasts: np.ndarray,
                               confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Perform Kupiec's Proportion of Failures (POF) test for VaR backtesting.
        
        Args:
            returns: Actual returns
            var_forecasts: VaR forecasts
            confidence_level: Confidence level used for VaR
            
        Returns:
            Backtesting results including test statistics
        """
        
        if len(returns) != len(var_forecasts):
            raise ValueError("Returns and VaR forecasts must have same length")
        
        # Convert returns to losses
        losses = -returns
        
        # Count violations (actual loss > VaR)
        violations = losses > var_forecasts
        num_violations = np.sum(violations)
        
        # Expected number of violations
        n = len(returns)
        p = 1 - confidence_level  # Expected violation rate
        expected_violations = n * p
        
        # Kupiec test statistic
        if num_violations == 0:
            lr_stat = 0
        else:
            observed_rate = num_violations / n
            lr_stat = -2 * np.log((p**num_violations * (1-p)**(n-num_violations)) / 
                                 (observed_rate**num_violations * (1-observed_rate)**(n-num_violations)))
        
        # Critical value (chi-squared with 1 degree of freedom)
        critical_value_95 = 3.841
        critical_value_99 = 6.635
        
        # P-value
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return {
            'num_observations': n,
            'num_violations': num_violations,
            'expected_violations': expected_violations,
            'violation_rate': num_violations / n,
            'expected_violation_rate': p,
            'lr_statistic': lr_stat,
            'p_value': p_value,
            'reject_at_95': lr_stat > critical_value_95,
            'reject_at_99': lr_stat > critical_value_99,
            'test_result': 'PASS' if lr_stat <= critical_value_95 else 'FAIL'
        }


class ExpectedShortfallCalculator:
    """Enhanced Expected Shortfall calculator."""
    
    def __init__(self):
        """Initialize Expected Shortfall calculator."""
        self.logger = get_logger(__name__)
    
    def calculate_expected_shortfall(self, 
                                   returns: np.ndarray,
                                   confidence_level: float = 0.95,
                                   method: str = "historical") -> Dict[str, Any]:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level
            method: Calculation method
            
        Returns:
            Expected Shortfall results
        """
        
        alpha = 1 - confidence_level
        
        if method == "historical":
            # Historical Expected Shortfall
            var_threshold = np.percentile(returns, alpha * 100)
            tail_returns = returns[returns <= var_threshold]
            es_value = -np.mean(tail_returns) if len(tail_returns) > 0 else 0
            
        elif method == "parametric":
            # Parametric Expected Shortfall (assuming normality)
            mean_return = np.mean(returns)
            volatility = np.std(returns)
            z_alpha = norm.ppf(alpha)
            es_value = -(mean_return + volatility * norm.pdf(z_alpha) / alpha)
            
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return {
            'expected_shortfall': es_value,
            'confidence_level': confidence_level,
            'method': method,
            'sample_size': len(returns)
        }
    
    def spectral_risk_measure(self, 
                             returns: np.ndarray,
                             risk_spectrum: str = "exponential",
                             parameter: float = 1.0) -> float:
        """
        Calculate spectral risk measure with different risk spectrums.
        
        Args:
            returns: Array of returns
            risk_spectrum: Type of risk spectrum ('exponential', 'power')
            parameter: Spectrum parameter
            
        Returns:
            Spectral risk measure value
        """
        
        # Sort returns (losses are negative)
        sorted_returns = np.sort(returns)
        n = len(returns)
        
        if risk_spectrum == "exponential":
            # Exponential spectrum: φ(u) = λe^(λu) for λ > 0
            u_values = np.arange(1, n + 1) / n
            weights = parameter * np.exp(parameter * u_values)
            weights = weights / np.sum(weights)  # Normalize
            
        elif risk_spectrum == "power":
            # Power spectrum: φ(u) = γu^(γ-1) for γ > 1
            u_values = np.arange(1, n + 1) / n
            weights = parameter * (u_values ** (parameter - 1))
            weights = weights / np.sum(weights)  # Normalize
            
        else:
            raise ValueError(f"Unknown risk spectrum: {risk_spectrum}")
        
        # Calculate weighted average
        spectral_measure = -np.sum(weights * sorted_returns)
        
        return spectral_measure
    
    def conditional_drawdown_at_risk(self, 
                                    returns: pd.Series,
                                    confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Calculate Conditional Drawdown at Risk (CDaR).
        
        Args:
            returns: Time series of returns
            confidence_level: Confidence level
            
        Returns:
            CDaR results
        """
        
        # Calculate cumulative returns and drawdowns
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Calculate DaR (Drawdown at Risk)
        alpha = 1 - confidence_level
        dar_threshold = np.percentile(drawdowns, alpha * 100)
        
        # Calculate CDaR (mean of tail drawdowns)
        tail_drawdowns = drawdowns[drawdowns <= dar_threshold]
        cdar_value = np.mean(tail_drawdowns) if len(tail_drawdowns) > 0 else 0
        
        return {
            'conditional_drawdown_at_risk': abs(cdar_value),
            'drawdown_at_risk': abs(dar_threshold),
            'max_drawdown': abs(np.min(drawdowns)),
            'confidence_level': confidence_level,
            'num_drawdown_periods': len(tail_drawdowns)
        }