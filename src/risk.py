from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from .options import OptionsCalculator, OptionType, StrategyDefinition


class StochasticProcess(Enum):
    GEOMETRIC_BROWNIAN_MOTION = "gbm"
    JUMP_DIFFUSION = "jump_diffusion"
    HESTON = "heston"
    HISTORICAL_BOOTSTRAP = "bootstrap"


@dataclass
class ProcessParameters:
    mu: float
    sigma: float
    jump_intensity: Optional[float] = None
    jump_mean: Optional[float] = None
    jump_std: Optional[float] = None
    kappa: Optional[float] = None
    theta: Optional[float] = None
    xi: Optional[float] = None
    rho: Optional[float] = None
    historical_returns: Optional[np.ndarray] = None


@dataclass
class RiskMetrics:
    mean_pnl: float
    std_pnl: float
    min_pnl: float
    max_pnl: float
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    probability_of_profit: float
    probability_of_max_loss: float
    skewness: float
    kurtosis: float
    profit_factor: float
    win_rate: float


class MonteCarloEngine:
    def __init__(self, num_simulations: int = 10000, random_seed: int = 42):
        self.num_simulations = num_simulations
        self.random_seed = random_seed
        self.calculator = OptionsCalculator()
        np.random.seed(self.random_seed)

    def simulate_price_paths(
        self,
        initial_price: float,
        process: StochasticProcess,
        parameters: ProcessParameters,
        time_horizon: float,
        num_steps: int,
        num_paths: Optional[int] = None,
    ) -> np.ndarray:
        num_paths = num_paths or self.num_simulations
        dt = time_horizon / num_steps

        if process == StochasticProcess.GEOMETRIC_BROWNIAN_MOTION:
            return self._simulate_gbm(initial_price, parameters, dt, num_steps, num_paths)
        elif process == StochasticProcess.JUMP_DIFFUSION:
            return self._simulate_jump_diffusion(initial_price, parameters, dt, num_steps, num_paths)
        elif process == StochasticProcess.HESTON:
            return self._simulate_heston(initial_price, parameters, dt, num_steps, num_paths)
        elif process == StochasticProcess.HISTORICAL_BOOTSTRAP:
            return self._simulate_bootstrap(initial_price, parameters, num_steps, num_paths)
        raise ValueError(f"Unsupported process: {process}")

    def _simulate_gbm(
        self,
        S0: float,
        params: ProcessParameters,
        dt: float,
        num_steps: int,
        num_paths: int,
    ) -> np.ndarray:
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = S0
        random_shocks = np.random.standard_normal((num_paths, num_steps))

        for t in range(1, num_steps + 1):
            drift = (params.mu - 0.5 * params.sigma**2) * dt
            diffusion = params.sigma * np.sqrt(dt) * random_shocks[:, t - 1]
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion)
        return paths

    def _simulate_jump_diffusion(
        self,
        S0: float,
        params: ProcessParameters,
        dt: float,
        num_steps: int,
        num_paths: int,
    ) -> np.ndarray:
        if params.jump_intensity is None:
            raise ValueError("Jump intensity required")
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = S0
        random_shocks = np.random.standard_normal((num_paths, num_steps))
        poisson_jumps = np.random.poisson(params.jump_intensity * dt, (num_paths, num_steps))

        for t in range(1, num_steps + 1):
            drift = (params.mu - 0.5 * params.sigma**2) * dt
            diffusion = params.sigma * np.sqrt(dt) * random_shocks[:, t - 1]
            jump_sizes = np.zeros(num_paths)
            jump_mask = poisson_jumps[:, t - 1] > 0
            if np.any(jump_mask):
                num_jumps = poisson_jumps[jump_mask, t - 1]
                for i, n_jumps in enumerate(num_jumps):
                    if n_jumps > 0:
                        jump_sizes[jump_mask][i] = np.sum(np.random.normal(params.jump_mean, params.jump_std, n_jumps))
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion + jump_sizes)
        return paths

    def _simulate_heston(
        self,
        S0: float,
        params: ProcessParameters,
        dt: float,
        num_steps: int,
        num_paths: int,
    ) -> np.ndarray:
        paths = np.zeros((num_paths, num_steps + 1))
        variance_paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = S0
        variance_paths[:, 0] = params.sigma**2
        z1 = np.random.standard_normal((num_paths, num_steps))
        z2 = np.random.standard_normal((num_paths, num_steps))
        w1 = z1
        w2 = params.rho * z1 + np.sqrt(1 - params.rho**2) * z2

        for t in range(1, num_steps + 1):
            dv = (
                params.kappa * (params.theta - variance_paths[:, t - 1]) * dt
                + params.xi * np.sqrt(np.maximum(variance_paths[:, t - 1], 0)) * np.sqrt(dt) * w2[:, t - 1]
            )
            variance_paths[:, t] = np.maximum(variance_paths[:, t - 1] + dv, 0)
            drift = params.mu * dt
            diffusion = np.sqrt(np.maximum(variance_paths[:, t - 1], 0)) * np.sqrt(dt) * w1[:, t - 1]
            paths[:, t] = paths[:, t - 1] * np.exp(drift - 0.5 * variance_paths[:, t - 1] * dt + diffusion)
        return paths

    def _simulate_bootstrap(self, S0: float, params: ProcessParameters, num_steps: int, num_paths: int) -> np.ndarray:
        if params.historical_returns is None:
            raise ValueError("Historical returns required")
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = S0
        for path in range(num_paths):
            for t in range(1, num_steps + 1):
                paths[path, t] = paths[path, t - 1] * (1 + np.random.choice(params.historical_returns))
        return paths

    def simulate_strategy_performance(
        self,
        strategy: StrategyDefinition,
        initial_price: float,
        process: StochasticProcess,
        parameters: ProcessParameters,
        time_horizon: Optional[float] = None,
    ) -> Dict[str, Any]:
        time_horizon = time_horizon or strategy.legs[0].time_to_expiry
        num_steps = max(int(time_horizon * 252), 30)
        price_paths = self.simulate_price_paths(initial_price, process, parameters, time_horizon, num_steps)
        final_prices = price_paths[:, -1]
        payoffs = self._calculate_strategy_payoffs(strategy, final_prices, initial_price, parameters.sigma)
        risk_metrics = self._calculate_risk_metrics(payoffs)

        return {
            "payoffs": payoffs,
            "final_prices": final_prices,
            "price_changes": (final_prices - initial_price) / initial_price,
            "risk_metrics": risk_metrics,
            "simulation_parameters": {
                "num_simulations": self.num_simulations,
                "time_horizon": time_horizon,
            },
        }

    def _calculate_strategy_payoffs(
        self,
        strategy: StrategyDefinition,
        final_prices: np.ndarray,
        initial_price: float,
        volatility: float,
    ) -> np.ndarray:
        initial_cost = sum(
            leg.position
            * leg.quantity
            * self.calculator.black_scholes_price(
                initial_price,
                leg.strike_price,
                leg.time_to_expiry,
                0.001,
                volatility,
                leg.option_type,
            )
            for leg in strategy.legs
        )

        payoffs = np.zeros(len(final_prices))
        for i, final_price in enumerate(final_prices):
            total_payoff = sum(
                leg.position
                * leg.quantity
                * (
                    max(final_price - leg.strike_price, 0)
                    if leg.option_type == OptionType.CALL
                    else max(leg.strike_price - final_price, 0)
                )
                for leg in strategy.legs
            )
            payoffs[i] = total_payoff - initial_cost
        return payoffs

    def _calculate_risk_metrics(self, payoffs: np.ndarray) -> RiskMetrics:
        mean_pnl = np.mean(payoffs)
        std_pnl = np.std(payoffs)
        var_95 = np.percentile(payoffs, 5)
        var_99 = np.percentile(payoffs, 1)

        pos = payoffs[payoffs > 0]
        neg = payoffs[payoffs < 0]
        avg_win = np.mean(pos) if len(pos) > 0 else 0
        avg_loss = np.mean(np.abs(neg)) if len(neg) > 0 else 0

        return RiskMetrics(
            mean_pnl=mean_pnl,
            std_pnl=std_pnl,
            min_pnl=np.min(payoffs),
            max_pnl=np.max(payoffs),
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=np.mean(payoffs[payoffs <= var_95]),
            expected_shortfall_99=np.mean(payoffs[payoffs <= var_99]),
            sharpe_ratio=mean_pnl / max(std_pnl, 0.001),
            sortino_ratio=mean_pnl / max(np.sqrt(np.mean(neg**2)) if len(neg) > 0 else 0.001, 0.001),
            max_drawdown=np.max(np.maximum.accumulate(np.cumsum(payoffs)) - np.cumsum(payoffs)),
            probability_of_profit=np.mean(payoffs > 0),
            probability_of_max_loss=np.mean(payoffs == np.min(payoffs)),
            skewness=stats.skew(payoffs),
            kurtosis=stats.kurtosis(payoffs),
            profit_factor=avg_win / max(avg_loss, 0.001),
            win_rate=len(pos) / len(payoffs),
        )

    def stress_test(
        self,
        strategy: StrategyDefinition,
        initial_price: float,
        base_volatility: float,
        stress_scenarios: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, Any]]:
        results = {}
        for name, params in stress_scenarios.items():
            stress_params = ProcessParameters(mu=params.get("mu", 0.0), sigma=params.get("sigma", base_volatility))
            results[name] = self.simulate_strategy_performance(
                strategy,
                initial_price,
                StochasticProcess.GEOMETRIC_BROWNIAN_MOTION,
                stress_params,
            )
        return results

    def scenario_analysis(
        self,
        strategy: StrategyDefinition,
        initial_price: float,
        volatility_range: Tuple[float, float],
        drift_range: Tuple[float, float],
        num_scenarios: int = 25,
    ) -> pd.DataFrame:
        results = []
        for vol in np.linspace(volatility_range[0], volatility_range[1], num_scenarios):
            for drift in np.linspace(drift_range[0], drift_range[1], num_scenarios):
                params = ProcessParameters(mu=drift, sigma=vol)
                res = self.simulate_strategy_performance(
                    strategy,
                    initial_price,
                    StochasticProcess.GEOMETRIC_BROWNIAN_MOTION,
                    params,
                )
                results.append(
                    {
                        "volatility": vol,
                        "drift": drift,
                        "mean_pnl": res["risk_metrics"].mean_pnl,
                        "var_95": res["risk_metrics"].var_95,
                        "sharpe_ratio": res["risk_metrics"].sharpe_ratio,
                    }
                )
        return pd.DataFrame(results)


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
