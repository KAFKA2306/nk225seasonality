from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from .config import JapaneseMarketConstants


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class GreeksResult:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    price: float


class OptionsCalculator:
    def __init__(self):
        self.default_risk_free_rate = JapaneseMarketConstants.DEFAULT_RISK_FREE_RATE
        self.trading_days_per_year = 252

    def black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> float:
        if T <= 0:
            return max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0)
        if sigma <= 0:
            return (
                max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
                if option_type == OptionType.CALL
                else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
            )

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == OptionType.CALL:
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    def monte_carlo_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        simulations: int = 100000,
    ) -> Tuple[float, float]:
        np.random.seed(42)
        z = np.random.standard_normal(simulations // 2)
        z = np.concatenate([z, -z])
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        payoffs = np.maximum(ST - K, 0) if option_type == OptionType.CALL else np.maximum(K - ST, 0)
        return np.exp(-r * T) * np.mean(payoffs), np.exp(-r * T) * np.std(payoffs) / np.sqrt(simulations)

    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> float:
        if market_price <= 0:
            return 0.0

        def price_diff(vol):
            try:
                return self.black_scholes_price(S, K, T, r, vol, option_type, q) - market_price
            except Exception:
                return float("inf")

        try:
            return brentq(price_diff, 0.001, 5.0, xtol=1e-6)
        except Exception:
            return np.nan


class GreeksCalculator:
    def __init__(self, calculator: OptionsCalculator):
        self.calculator = calculator

    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> GreeksResult:
        if T <= 0:
            return GreeksResult(
                0,
                0,
                0,
                0,
                0,
                max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0),
            )
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        price = self.calculator.black_scholes_price(S, K, T, r, sigma, option_type, q)
        delta = np.exp(-q * T) * norm.cdf(d1) if option_type == OptionType.CALL else -np.exp(-q * T) * norm.cdf(-d1)
        gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100

        if option_type == OptionType.CALL:
            theta = (
                (-S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
                + q * S * np.exp(-q * T) * norm.cdf(d1)
            ) / 365.25
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            theta = (
                (-S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
                - q * S * np.exp(-q * T) * norm.cdf(-d1)
            ) / 365.25
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return GreeksResult(delta, gamma, theta, vega, rho, price)


class StrategyType(Enum):
    PUT_SPREAD = "put_spread"
    CALL_SPREAD = "call_spread"
    STRADDLE = "straddle"


@dataclass
class StrategyLeg:
    option_type: OptionType
    strike_price: float
    time_to_expiry: float
    position: int
    quantity: int = 1


@dataclass
class StrategyDefinition:
    strategy_type: StrategyType
    legs: List[StrategyLeg]
    target_month: int
    entry_days_before: int
    exit_days_before: int
    max_loss: Optional[float] = None


class SeasonalOptionsStrategy:
    def __init__(self, market_data: pd.DataFrame, seasonality_results: Dict[int, Any]):
        self.market_data = market_data
        self.seasonality = seasonality_results
        self.calculator = OptionsCalculator()
        self.days_to_expiry = 30

    def design_put_spread_strategy(self, target_month: int, confidence_level: float = 0.8) -> StrategyDefinition:
        if target_month not in self.seasonality:
            raise ValueError(f"No seasonality for {target_month}")
        stats = self.seasonality[target_month]
        current_price = self.market_data["close_price"].iloc[-1] if "close_price" in self.market_data.columns else 100
        move = abs(stats.mean_return)
        adj = stats.std_return * confidence_level
        long_strike = max(current_price * (1 - move - adj), current_price * 0.8)
        short_strike = min(current_price * (1 - move), current_price * 0.95)

        legs = [
            StrategyLeg(OptionType.PUT, long_strike, self.days_to_expiry / 365.25, 1),
            StrategyLeg(OptionType.PUT, short_strike, self.days_to_expiry / 365.25, -1),
        ]
        return StrategyDefinition(
            StrategyType.PUT_SPREAD,
            legs,
            target_month,
            5,
            5,
            short_strike - long_strike,
        )

    def design_call_spread_strategy(self, target_month: int, confidence_level: float = 0.8) -> StrategyDefinition:
        if target_month not in self.seasonality:
            raise ValueError(f"No seasonality for {target_month}")
        stats = self.seasonality[target_month]
        current_price = self.market_data["close_price"].iloc[-1] if "close_price" in self.market_data.columns else 100
        move = stats.mean_return
        adj = stats.std_return * confidence_level
        long_strike = max(current_price * (1 + move - adj), current_price * 1.02)
        short_strike = min(current_price * (1 + move + adj), current_price * 1.15)

        legs = [
            StrategyLeg(OptionType.CALL, long_strike, self.days_to_expiry / 365.25, 1),
            StrategyLeg(OptionType.CALL, short_strike, self.days_to_expiry / 365.25, -1),
        ]
        return StrategyDefinition(StrategyType.CALL_SPREAD, legs, target_month, 5, 5)

    def calculate_strategy_payoff(
        self,
        strategy: StrategyDefinition,
        underlying_prices: np.ndarray,
        volatility: float,
    ) -> Dict[str, Any]:
        payoffs = np.zeros(len(underlying_prices))
        initial_cost = sum(
            leg.position
            * leg.quantity
            * self.calculator.black_scholes_price(
                self.market_data["close_price"].iloc[-1],
                leg.strike_price,
                leg.time_to_expiry,
                JapaneseMarketConstants.DEFAULT_RISK_FREE_RATE,
                volatility,
                leg.option_type,
            )
            for leg in strategy.legs
        )

        for i, S in enumerate(underlying_prices):
            total = sum(
                leg.position
                * leg.quantity
                * (max(S - leg.strike_price, 0) if leg.option_type == OptionType.CALL else max(leg.strike_price - S, 0))
                for leg in strategy.legs
            )
            payoffs[i] = total - initial_cost

        return {
            "net_pnl": payoffs,
            "max_profit": np.max(payoffs),
            "max_loss": np.min(payoffs),
        }


class StrategyBacktester:
    def __init__(self, market_data: pd.DataFrame):
        self.market_data = market_data
        self.calculator = OptionsCalculator()

    def backtest_strategy(
        self, strategy: StrategyDefinition, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        data = self.market_data[(self.market_data.index >= start_date) & (self.market_data.index <= end_date)].copy()
        if data.empty:
            return {"error": "No data"}

        return {"total_return": 0.0, "sharpe_ratio": 0.0}
