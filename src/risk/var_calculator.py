from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm


class VaRMethod(Enum):
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"
    EXTREME_VALUE = "extreme_value"


@dataclass
class VaRResult:
    confidence_level: float
    var_value: float
    expected_shortfall: float
    method: str
    sample_size: int
    volatility: float
    mean_return: float


class VaRCalculator:
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        method: VaRMethod = VaRMethod.HISTORICAL,
        holding_period: int = 1,
    ) -> VaRResult:
        if len(returns) == 0:
            raise ValueError("Returns empty")
        scaled_returns = returns * np.sqrt(holding_period) if holding_period > 1 else returns

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
        raise ValueError(f"Unsupported method: {method}")

    def _parametric_var(self, returns: np.ndarray, confidence_level: float) -> VaRResult:
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        alpha = 1 - confidence_level
        z_score = norm.ppf(alpha)
        var_value = -(mean_return + z_score * volatility)
        expected_shortfall = -(mean_return + volatility * norm.pdf(z_score) / alpha)
        return VaRResult(
            confidence_level,
            var_value,
            expected_shortfall,
            "parametric_normal",
            len(returns),
            volatility,
            mean_return,
        )

    def _historical_var(self, returns: np.ndarray, confidence_level: float) -> VaRResult:
        alpha = 1 - confidence_level
        var_value = -np.percentile(returns, alpha * 100)
        tail_returns = returns[returns <= -var_value]
        expected_shortfall = -np.mean(tail_returns) if len(tail_returns) > 0 else 0
        return VaRResult(
            confidence_level,
            var_value,
            expected_shortfall,
            "historical",
            len(returns),
            np.std(returns),
            np.mean(returns),
        )

    def _monte_carlo_var(self, returns: np.ndarray, confidence_level: float, num_simulations: int = 10000) -> VaRResult:
        np.random.seed(42)
        sim_returns = np.random.choice(returns, size=num_simulations, replace=True)
        alpha = 1 - confidence_level
        var_value = -np.percentile(sim_returns, alpha * 100)
        tail_returns = sim_returns[sim_returns <= -var_value]
        return VaRResult(
            confidence_level,
            var_value,
            -np.mean(tail_returns),
            "monte_carlo",
            len(returns),
            np.std(returns),
            np.mean(returns),
        )

    def _cornish_fisher_var(self, returns: np.ndarray, confidence_level: float) -> VaRResult:
        mean, vol, skew, kurt = (
            np.mean(returns),
            np.std(returns),
            stats.skew(returns),
            stats.kurtosis(returns),
        )
        alpha = 1 - confidence_level
        z = norm.ppf(alpha)
        cf_q = z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * kurt / 24 - (2 * z**3 - 5 * z) * skew**2 / 36
        var_value = -(mean + cf_q * vol)
        expected_shortfall = -(
            mean + vol * norm.pdf(z) / alpha * (1 + skew * (z**2 - 1) / 6 + kurt * z * (z**2 - 3) / 24)
        )
        return VaRResult(
            confidence_level,
            var_value,
            expected_shortfall,
            "cornish_fisher",
            len(returns),
            vol,
            mean,
        )

    def _extreme_value_var(self, returns: np.ndarray, confidence_level: float) -> VaRResult:
        threshold = np.percentile(-returns, 90)
        excesses = (-returns)[(-returns) > threshold] - threshold
        if len(excesses) < 10:
            return self._historical_var(returns, confidence_level)

        try:
            from scipy.stats import genpareto

            shape, _, scale = genpareto.fit(excesses, floc=0)
            n, nu, alpha = len(returns), len(excesses), 1 - confidence_level
            var_value = (
                threshold + (scale / shape) * (((n / nu) * alpha) ** (-shape) - 1)
                if shape != 0
                else threshold + scale * np.log((n / nu) * alpha)
            )
            es = (
                (var_value + scale - shape * threshold) / (1 - shape)
                if shape < 1 and shape != 0
                else -np.mean(returns[returns <= -var_value])
            )
            return VaRResult(
                confidence_level,
                var_value,
                es,
                "evt",
                len(returns),
                np.std(returns),
                np.mean(returns),
            )
        except Exception:
            return self._historical_var(returns, confidence_level)

    def rolling_var(
        self,
        returns: pd.Series,
        window: int = 252,
        confidence_level: float = 0.95,
        method: VaRMethod = VaRMethod.HISTORICAL,
    ) -> pd.Series:
        vars = [
            self.calculate_var(returns.iloc[i - window : i].values, confidence_level, method).var_value
            for i in range(window, len(returns))
        ]
        return pd.Series(vars, index=returns.index[window:], name=f"VaR_{confidence_level}")

    def backtesting_kupiec_test(
        self,
        returns: np.ndarray,
        var_forecasts: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        if len(returns) != len(var_forecasts):
            raise ValueError("Length mismatch")
        violations = (-returns) > var_forecasts
        num_v = np.sum(violations)
        n, p = len(returns), 1 - confidence_level
        obs_rate = num_v / n
        lr_stat = (
            -2 * np.log((p**num_v * (1 - p) ** (n - num_v)) / (obs_rate**num_v * (1 - obs_rate) ** (n - num_v)))
            if num_v > 0
            else 0
        )
        return {
            "num_violations": num_v,
            "violation_rate": obs_rate,
            "p_value": 1 - stats.chi2.cdf(lr_stat, df=1),
            "test_result": "PASS" if lr_stat <= 3.841 else "FAIL",
        }


class ExpectedShortfallCalculator:
    def calculate_expected_shortfall(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> Dict[str, Any]:
        alpha = 1 - confidence_level
        if method == "historical":
            tail = returns[returns <= np.percentile(returns, alpha * 100)]
            es = -np.mean(tail) if len(tail) > 0 else 0
        elif method == "parametric":
            es = -(np.mean(returns) + np.std(returns) * norm.pdf(norm.ppf(alpha)) / alpha)
        else:
            raise ValueError(f"Unsupported method: {method}")
        return {"expected_shortfall": es, "confidence_level": confidence_level}
