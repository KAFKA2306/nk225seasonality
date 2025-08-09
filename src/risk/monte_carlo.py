"""
Monte Carlo simulation engine for risk assessment and strategy validation.

This module provides comprehensive Monte Carlo simulation capabilities for
options strategies, including multiple stochastic processes and variance
reduction techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
from scipy import stats

from ..config import get_logger
from ..options.strategies import StrategyDefinition, StrategyLeg
from ..options.calculator import OptionsCalculator, OptionType


class StochasticProcess(Enum):
    """Types of stochastic processes for simulation."""
    GEOMETRIC_BROWNIAN_MOTION = "gbm"
    JUMP_DIFFUSION = "jump_diffusion"
    HESTON = "heston"
    HISTORICAL_BOOTSTRAP = "bootstrap"


@dataclass
class ProcessParameters:
    """Parameters for stochastic processes."""
    
    # Common parameters
    mu: float  # Drift
    sigma: float  # Volatility
    
    # Jump diffusion parameters
    jump_intensity: Optional[float] = None
    jump_mean: Optional[float] = None
    jump_std: Optional[float] = None
    
    # Heston parameters
    kappa: Optional[float] = None  # Mean reversion speed
    theta: Optional[float] = None  # Long-term variance
    xi: Optional[float] = None     # Volatility of volatility
    rho: Optional[float] = None    # Correlation
    
    # Bootstrap parameters
    historical_returns: Optional[np.ndarray] = None


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    
    # Basic statistics
    mean_pnl: float
    std_pnl: float
    min_pnl: float
    max_pnl: float
    
    # Risk measures
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    
    # Performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    
    # Probability metrics
    probability_of_profit: float
    probability_of_max_loss: float
    
    # Distribution characteristics
    skewness: float
    kurtosis: float
    
    # Additional metrics
    profit_factor: float
    win_rate: float


class MonteCarloEngine:
    """Monte Carlo simulation engine for risk assessment."""
    
    def __init__(self, num_simulations: int = 10000, random_seed: int = 42):
        """
        Initialize Monte Carlo engine.
        
        Args:
            num_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        """
        self.num_simulations = num_simulations
        self.random_seed = random_seed
        self.logger = get_logger(__name__)
        self.calculator = OptionsCalculator()
        
        # Set random seed
        np.random.seed(self.random_seed)
    
    def simulate_price_paths(self, 
                           initial_price: float,
                           process: StochasticProcess,
                           parameters: ProcessParameters,
                           time_horizon: float,
                           num_steps: int,
                           num_paths: Optional[int] = None) -> np.ndarray:
        """
        Simulate price paths using specified stochastic process.
        
        Args:
            initial_price: Starting price
            process: Type of stochastic process
            parameters: Process parameters
            time_horizon: Time horizon in years
            num_steps: Number of time steps
            num_paths: Number of paths (defaults to num_simulations)
            
        Returns:
            Array of simulated price paths (num_paths x num_steps+1)
        """
        if num_paths is None:
            num_paths = self.num_simulations
        
        dt = time_horizon / num_steps
        
        if process == StochasticProcess.GEOMETRIC_BROWNIAN_MOTION:
            return self._simulate_gbm(initial_price, parameters, dt, num_steps, num_paths)
        
        elif process == StochasticProcess.JUMP_DIFFUSION:
            return self._simulate_jump_diffusion(initial_price, parameters, dt, num_steps, num_paths)
        
        elif process == StochasticProcess.HESTON:
            return self._simulate_heston(initial_price, parameters, dt, num_steps, num_paths)
        
        elif process == StochasticProcess.HISTORICAL_BOOTSTRAP:
            return self._simulate_bootstrap(initial_price, parameters, num_steps, num_paths)
        
        else:
            raise ValueError(f"Unsupported process: {process}")
    
    def _simulate_gbm(self, 
                     S0: float, 
                     params: ProcessParameters,
                     dt: float, 
                     num_steps: int, 
                     num_paths: int) -> np.ndarray:
        """Simulate Geometric Brownian Motion paths."""
        
        # Pre-generate all random numbers
        random_shocks = np.random.standard_normal((num_paths, num_steps))
        
        # Initialize price paths
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = S0
        
        # Simulate paths
        for t in range(1, num_steps + 1):
            drift = (params.mu - 0.5 * params.sigma**2) * dt
            diffusion = params.sigma * np.sqrt(dt) * random_shocks[:, t-1]
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion)
        
        return paths
    
    def _simulate_jump_diffusion(self, 
                                S0: float,
                                params: ProcessParameters,
                                dt: float,
                                num_steps: int,
                                num_paths: int) -> np.ndarray:
        """Simulate Jump Diffusion (Merton) model paths."""
        
        if params.jump_intensity is None:
            raise ValueError("Jump intensity required for jump diffusion")
        
        # Pre-generate random numbers
        random_shocks = np.random.standard_normal((num_paths, num_steps))
        poisson_jumps = np.random.poisson(params.jump_intensity * dt, (num_paths, num_steps))
        
        # Initialize paths
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = S0
        
        for t in range(1, num_steps + 1):
            # Diffusion component
            drift = (params.mu - 0.5 * params.sigma**2) * dt
            diffusion = params.sigma * np.sqrt(dt) * random_shocks[:, t-1]
            
            # Jump component
            jump_sizes = np.zeros(num_paths)
            jump_mask = poisson_jumps[:, t-1] > 0
            
            if np.any(jump_mask):
                num_jumps = poisson_jumps[jump_mask, t-1]
                for i, n_jumps in enumerate(num_jumps):
                    if n_jumps > 0:
                        jump_returns = np.random.normal(
                            params.jump_mean, params.jump_std, n_jumps
                        )
                        jump_sizes[jump_mask][i] = np.sum(jump_returns)
            
            # Update paths
            log_returns = drift + diffusion + jump_sizes
            paths[:, t] = paths[:, t-1] * np.exp(log_returns)
        
        return paths
    
    def _simulate_heston(self, 
                        S0: float,
                        params: ProcessParameters,
                        dt: float,
                        num_steps: int,
                        num_paths: int) -> np.ndarray:
        """Simulate Heston stochastic volatility model paths."""
        
        if any(p is None for p in [params.kappa, params.theta, params.xi, params.rho]):
            raise ValueError("All Heston parameters required")
        
        # Initialize arrays
        paths = np.zeros((num_paths, num_steps + 1))
        variance_paths = np.zeros((num_paths, num_steps + 1))
        
        paths[:, 0] = S0
        variance_paths[:, 0] = params.sigma**2  # Initial variance
        
        # Pre-generate correlated random numbers
        z1 = np.random.standard_normal((num_paths, num_steps))
        z2 = np.random.standard_normal((num_paths, num_steps))
        
        # Apply correlation
        w1 = z1
        w2 = params.rho * z1 + np.sqrt(1 - params.rho**2) * z2
        
        for t in range(1, num_steps + 1):
            # Update variance using Euler scheme
            dv = (params.kappa * (params.theta - variance_paths[:, t-1]) * dt + 
                  params.xi * np.sqrt(np.maximum(variance_paths[:, t-1], 0)) * 
                  np.sqrt(dt) * w2[:, t-1])
            
            variance_paths[:, t] = np.maximum(variance_paths[:, t-1] + dv, 0)
            
            # Update stock price
            drift = params.mu * dt
            diffusion = (np.sqrt(np.maximum(variance_paths[:, t-1], 0)) * 
                        np.sqrt(dt) * w1[:, t-1])
            
            paths[:, t] = paths[:, t-1] * np.exp(drift - 0.5 * variance_paths[:, t-1] * dt + diffusion)
        
        return paths
    
    def _simulate_bootstrap(self, 
                           S0: float,
                           params: ProcessParameters,
                           num_steps: int,
                           num_paths: int) -> np.ndarray:
        """Simulate paths using historical bootstrap."""
        
        if params.historical_returns is None:
            raise ValueError("Historical returns required for bootstrap")
        
        historical_returns = params.historical_returns
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = S0
        
        for path in range(num_paths):
            for t in range(1, num_steps + 1):
                # Randomly sample from historical returns
                random_return = np.random.choice(historical_returns)
                paths[path, t] = paths[path, t-1] * (1 + random_return)
        
        return paths
    
    def simulate_strategy_performance(self, 
                                    strategy: StrategyDefinition,
                                    initial_price: float,
                                    process: StochasticProcess,
                                    parameters: ProcessParameters,
                                    time_horizon: Optional[float] = None) -> Dict[str, Any]:
        """
        Simulate strategy performance using Monte Carlo.
        
        Args:
            strategy: Options strategy to simulate
            initial_price: Initial underlying price
            process: Stochastic process to use
            parameters: Process parameters
            time_horizon: Time horizon (defaults to strategy expiry)
            
        Returns:
            Dictionary with simulation results and risk metrics
        """
        
        # Use strategy expiry as default time horizon
        if time_horizon is None:
            time_horizon = strategy.legs[0].time_to_expiry
        
        # Simulate price paths
        num_steps = max(int(time_horizon * 252), 30)  # At least 30 steps
        price_paths = self.simulate_price_paths(
            initial_price, process, parameters, time_horizon, num_steps
        )
        
        # Calculate strategy payoffs
        final_prices = price_paths[:, -1]
        payoffs = self._calculate_strategy_payoffs(strategy, final_prices, initial_price, parameters.sigma)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(payoffs)
        
        # Additional analysis
        price_changes = (final_prices - initial_price) / initial_price
        
        results = {
            'payoffs': payoffs,
            'final_prices': final_prices,
            'price_changes': price_changes,
            'risk_metrics': risk_metrics,
            'simulation_parameters': {
                'num_simulations': self.num_simulations,
                'time_horizon': time_horizon,
                'initial_price': initial_price,
                'process': process.value,
                'num_steps': num_steps
            }
        }
        
        return results
    
    def _calculate_strategy_payoffs(self, 
                                   strategy: StrategyDefinition,
                                   final_prices: np.ndarray,
                                   initial_price: float,
                                   volatility: float) -> np.ndarray:
        """Calculate strategy payoffs for each simulation."""
        
        payoffs = np.zeros(len(final_prices))
        
        # Calculate initial cost/credit
        initial_cost = 0.0
        for leg in strategy.legs:
            option_price = self.calculator.black_scholes_price(
                initial_price, leg.strike_price, leg.time_to_expiry,
                0.001, volatility, leg.option_type  # Using low risk-free rate for Japan
            )
            initial_cost += leg.position * leg.quantity * option_price
        
        # Calculate payoffs at expiration
        for i, final_price in enumerate(final_prices):
            total_payoff = 0.0
            
            for leg in strategy.legs:
                if leg.option_type == OptionType.CALL:
                    intrinsic_value = max(final_price - leg.strike_price, 0)
                else:
                    intrinsic_value = max(leg.strike_price - final_price, 0)
                
                total_payoff += leg.position * leg.quantity * intrinsic_value
            
            # Net P&L = payoff - initial cost
            payoffs[i] = total_payoff - initial_cost
        
        return payoffs
    
    def _calculate_risk_metrics(self, payoffs: np.ndarray) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        
        # Basic statistics
        mean_pnl = np.mean(payoffs)
        std_pnl = np.std(payoffs)
        min_pnl = np.min(payoffs)
        max_pnl = np.max(payoffs)
        
        # Risk measures
        var_95 = np.percentile(payoffs, 5)
        var_99 = np.percentile(payoffs, 1)
        
        # Expected shortfall (conditional VaR)
        es_95 = np.mean(payoffs[payoffs <= var_95])
        es_99 = np.mean(payoffs[payoffs <= var_99])
        
        # Performance ratios
        sharpe_ratio = mean_pnl / max(std_pnl, 0.001)
        
        # Sortino ratio (using downside deviation)
        negative_returns = payoffs[payoffs < 0]
        downside_deviation = np.sqrt(np.mean(negative_returns**2)) if len(negative_returns) > 0 else 0.001
        sortino_ratio = mean_pnl / max(downside_deviation, 0.001)
        
        # Maximum drawdown (simplified for single period)
        cumulative_returns = np.cumsum(payoffs)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns)
        
        # Probability metrics
        probability_of_profit = np.mean(payoffs > 0)
        probability_of_max_loss = np.mean(payoffs == min_pnl)
        
        # Distribution characteristics
        skewness = stats.skew(payoffs)
        kurtosis_val = stats.kurtosis(payoffs)
        
        # Additional metrics
        positive_payoffs = payoffs[payoffs > 0]
        negative_payoffs = payoffs[payoffs < 0]
        
        avg_win = np.mean(positive_payoffs) if len(positive_payoffs) > 0 else 0
        avg_loss = np.mean(np.abs(negative_payoffs)) if len(negative_payoffs) > 0 else 0
        profit_factor = avg_win / max(avg_loss, 0.001)
        
        win_rate = len(positive_payoffs) / len(payoffs)
        
        return RiskMetrics(
            mean_pnl=mean_pnl,
            std_pnl=std_pnl,
            min_pnl=min_pnl,
            max_pnl=max_pnl,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            probability_of_profit=probability_of_profit,
            probability_of_max_loss=probability_of_max_loss,
            skewness=skewness,
            kurtosis=kurtosis_val,
            profit_factor=profit_factor,
            win_rate=win_rate
        )
    
    def stress_test(self, 
                   strategy: StrategyDefinition,
                   initial_price: float,
                   base_volatility: float,
                   stress_scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """
        Perform stress testing under different market scenarios.
        
        Args:
            strategy: Options strategy to test
            initial_price: Initial underlying price
            base_volatility: Base volatility assumption
            stress_scenarios: Dictionary of stress scenarios
            
        Returns:
            Results for each stress scenario
        """
        
        results = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            self.logger.info(f"Running stress test scenario: {scenario_name}")
            
            # Create modified parameters for this scenario
            stress_params = ProcessParameters(
                mu=scenario_params.get('mu', 0.0),
                sigma=scenario_params.get('sigma', base_volatility)
            )
            
            # Run simulation for this scenario
            scenario_results = self.simulate_strategy_performance(
                strategy, initial_price, StochasticProcess.GEOMETRIC_BROWNIAN_MOTION, 
                stress_params
            )
            
            results[scenario_name] = scenario_results
        
        return results
    
    def parallel_simulation(self, 
                           strategy: StrategyDefinition,
                           initial_price: float,
                           process: StochasticProcess,
                           parameters: ProcessParameters,
                           num_workers: int = 4) -> Dict[str, Any]:
        """
        Run parallel Monte Carlo simulation for faster execution.
        
        Args:
            strategy: Options strategy to simulate
            initial_price: Initial underlying price
            process: Stochastic process to use
            parameters: Process parameters
            num_workers: Number of parallel workers
            
        Returns:
            Aggregated simulation results
        """
        
        sims_per_worker = self.num_simulations // num_workers
        
        def run_simulation_chunk(chunk_size):
            # Create temporary engine for this chunk
            temp_engine = MonteCarloEngine(chunk_size, self.random_seed)
            return temp_engine.simulate_strategy_performance(
                strategy, initial_price, process, parameters
            )
        
        # Run simulations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_chunk = {
                executor.submit(run_simulation_chunk, sims_per_worker): i 
                for i in range(num_workers)
            }
            
            chunk_results = []
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_results.append(future.result())
        
        # Aggregate results
        all_payoffs = np.concatenate([result['payoffs'] for result in chunk_results])
        all_final_prices = np.concatenate([result['final_prices'] for result in chunk_results])
        all_price_changes = np.concatenate([result['price_changes'] for result in chunk_results])
        
        # Recalculate risk metrics on aggregated data
        aggregated_risk_metrics = self._calculate_risk_metrics(all_payoffs)
        
        return {
            'payoffs': all_payoffs,
            'final_prices': all_final_prices,
            'price_changes': all_price_changes,
            'risk_metrics': aggregated_risk_metrics,
            'simulation_parameters': chunk_results[0]['simulation_parameters']
        }
    
    def scenario_analysis(self, 
                         strategy: StrategyDefinition,
                         initial_price: float,
                         volatility_range: Tuple[float, float],
                         drift_range: Tuple[float, float],
                         num_scenarios: int = 25) -> pd.DataFrame:
        """
        Perform scenario analysis across parameter ranges.
        
        Args:
            strategy: Options strategy to analyze
            initial_price: Initial underlying price
            volatility_range: Range of volatilities to test
            drift_range: Range of drifts to test
            num_scenarios: Number of scenarios per dimension
            
        Returns:
            DataFrame with scenario results
        """
        
        vol_values = np.linspace(volatility_range[0], volatility_range[1], num_scenarios)
        drift_values = np.linspace(drift_range[0], drift_range[1], num_scenarios)
        
        scenario_results = []
        
        for vol in vol_values:
            for drift in drift_values:
                params = ProcessParameters(mu=drift, sigma=vol)
                
                # Run smaller simulation for each scenario
                temp_engine = MonteCarloEngine(1000, self.random_seed)
                result = temp_engine.simulate_strategy_performance(
                    strategy, initial_price, StochasticProcess.GEOMETRIC_BROWNIAN_MOTION, params
                )
                
                scenario_results.append({
                    'volatility': vol,
                    'drift': drift,
                    'mean_pnl': result['risk_metrics'].mean_pnl,
                    'std_pnl': result['risk_metrics'].std_pnl,
                    'var_95': result['risk_metrics'].var_95,
                    'probability_of_profit': result['risk_metrics'].probability_of_profit,
                    'sharpe_ratio': result['risk_metrics'].sharpe_ratio
                })
        
        return pd.DataFrame(scenario_results)