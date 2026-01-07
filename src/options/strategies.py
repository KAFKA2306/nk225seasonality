from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import JapaneseMarketConstants
from .calculator import OptionsCalculator, OptionType


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
