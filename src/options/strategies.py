"""
Seasonal options strategies implementation.

This module provides strategy development, optimization, and backtesting
capabilities for seasonal options strategies on the Nikkei 225.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .calculator import OptionsCalculator, GreeksCalculator, OptionType
from ..config import get_logger, JapaneseMarketConstants


class StrategyType(Enum):
    """Types of options strategies."""
    PUT_SPREAD = "put_spread"
    CALL_SPREAD = "call_spread"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    IRON_CONDOR = "iron_condor"
    CALENDAR_SPREAD = "calendar_spread"


@dataclass
class StrategyLeg:
    """Individual leg of an options strategy."""
    
    option_type: OptionType
    strike_price: float
    time_to_expiry: float
    position: int  # +1 for long, -1 for short
    quantity: int = 1


@dataclass
class StrategyDefinition:
    """Complete strategy definition."""
    
    strategy_type: StrategyType
    legs: List[StrategyLeg]
    target_month: int
    entry_days_before: int
    exit_days_before: int
    max_loss: Optional[float] = None
    profit_target: Optional[float] = None


@dataclass
class StrategyResult:
    """Result of strategy analysis."""
    
    strategy_definition: StrategyDefinition
    expected_profit: float
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    probability_of_profit: float
    expected_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float


class SeasonalOptionsStrategy:
    """Framework for seasonal options strategies."""
    
    def __init__(self, 
                 market_data: pd.DataFrame, 
                 seasonality_results: Dict[int, Any]):
        """
        Initialize seasonal options strategy framework.
        
        Args:
            market_data: Historical market data
            seasonality_results: Results from seasonality analysis
        """
        self.market_data = market_data
        self.seasonality = seasonality_results
        self.calculator = OptionsCalculator()
        self.greeks_calculator = GreeksCalculator(self.calculator)
        self.logger = get_logger(__name__)
        
        # Default strategy parameters
        self.default_risk_free_rate = JapaneseMarketConstants.DEFAULT_RISK_FREE_RATE
        self.days_to_expiry = 30
        
    def design_put_spread_strategy(self, 
                                  target_month: int,
                                  confidence_level: float = 0.8) -> StrategyDefinition:
        """
        Design put spread strategy for seasonal patterns.
        
        Args:
            target_month: Month to target for the strategy
            confidence_level: Confidence level for strike selection
            
        Returns:
            Strategy definition
        """
        if target_month not in self.seasonality:
            raise ValueError(f"No seasonality data for month {target_month}")
        
        month_stats = self.seasonality[target_month]
        expected_return = month_stats.mean_return
        volatility = month_stats.std_return
        
        # Only create put spread if expecting negative returns
        if expected_return >= 0:
            self.logger.warning(f"Month {target_month} has positive expected return. Put spread may not be optimal.")
        
        # Current price (use last available)
        current_price = self.market_data['close_price'].iloc[-1] if 'close_price' in self.market_data.columns else 100
        
        # Calculate strikes based on expected move and confidence level
        expected_move = abs(expected_return)
        volatility_adjustment = volatility * confidence_level
        
        # Long put strike (further OTM)
        long_strike = current_price * (1 - expected_move - volatility_adjustment)
        
        # Short put strike (closer to ATM)  
        short_strike = current_price * (1 - expected_move)
        
        # Ensure reasonable strikes
        long_strike = max(long_strike, current_price * 0.8)   # Not too far OTM
        short_strike = min(short_strike, current_price * 0.95)  # Not too close to ATM
        
        # Create strategy legs
        legs = [
            StrategyLeg(
                option_type=OptionType.PUT,
                strike_price=long_strike,
                time_to_expiry=self.days_to_expiry / 365.25,
                position=1,  # Long
                quantity=1
            ),
            StrategyLeg(
                option_type=OptionType.PUT,
                strike_price=short_strike,
                time_to_expiry=self.days_to_expiry / 365.25,
                position=-1,  # Short
                quantity=1
            )
        ]
        
        return StrategyDefinition(
            strategy_type=StrategyType.PUT_SPREAD,
            legs=legs,
            target_month=target_month,
            entry_days_before=5,  # Enter 5 days before month start
            exit_days_before=5,   # Exit 5 days before expiry
            max_loss=short_strike - long_strike  # Maximum spread width
        )
    
    def design_call_spread_strategy(self, 
                                   target_month: int,
                                   confidence_level: float = 0.8) -> StrategyDefinition:
        """Design call spread strategy for positive seasonal patterns."""
        
        if target_month not in self.seasonality:
            raise ValueError(f"No seasonality data for month {target_month}")
        
        month_stats = self.seasonality[target_month]
        expected_return = month_stats.mean_return
        volatility = month_stats.std_return
        
        # Only create call spread if expecting positive returns
        if expected_return <= 0:
            self.logger.warning(f"Month {target_month} has negative expected return. Call spread may not be optimal.")
        
        current_price = self.market_data['close_price'].iloc[-1] if 'close_price' in self.market_data.columns else 100
        
        # Calculate strikes
        expected_move = expected_return
        volatility_adjustment = volatility * confidence_level
        
        # Long call strike (closer to ATM)
        long_strike = current_price * (1 + expected_move - volatility_adjustment)
        
        # Short call strike (further OTM)
        short_strike = current_price * (1 + expected_move + volatility_adjustment)
        
        # Ensure reasonable strikes
        long_strike = max(long_strike, current_price * 1.02)   # Slightly OTM
        short_strike = min(short_strike, current_price * 1.15)  # Not too far OTM
        
        legs = [
            StrategyLeg(
                option_type=OptionType.CALL,
                strike_price=long_strike,
                time_to_expiry=self.days_to_expiry / 365.25,
                position=1,  # Long
                quantity=1
            ),
            StrategyLeg(
                option_type=OptionType.CALL,
                strike_price=short_strike,
                time_to_expiry=self.days_to_expiry / 365.25,
                position=-1,  # Short
                quantity=1
            )
        ]
        
        return StrategyDefinition(
            strategy_type=StrategyType.CALL_SPREAD,
            legs=legs,
            target_month=target_month,
            entry_days_before=5,
            exit_days_before=5,
            max_loss=None  # Will be calculated as net premium paid
        )
    
    def design_straddle_strategy(self, 
                                target_month: int) -> StrategyDefinition:
        """Design straddle strategy for high volatility months."""
        
        if target_month not in self.seasonality:
            raise ValueError(f"No seasonality data for month {target_month}")
        
        month_stats = self.seasonality[target_month]
        volatility = month_stats.std_return
        
        # Straddles work best when expecting high volatility regardless of direction
        current_price = self.market_data['close_price'].iloc[-1] if 'close_price' in self.market_data.columns else 100
        
        # ATM straddle
        strike_price = current_price
        
        legs = [
            StrategyLeg(
                option_type=OptionType.CALL,
                strike_price=strike_price,
                time_to_expiry=self.days_to_expiry / 365.25,
                position=1,  # Long
                quantity=1
            ),
            StrategyLeg(
                option_type=OptionType.PUT,
                strike_price=strike_price,
                time_to_expiry=self.days_to_expiry / 365.25,
                position=1,  # Long
                quantity=1
            )
        ]
        
        return StrategyDefinition(
            strategy_type=StrategyType.STRADDLE,
            legs=legs,
            target_month=target_month,
            entry_days_before=10,  # Enter earlier for volatility plays
            exit_days_before=5,
            max_loss=None  # Premium paid
        )
    
    def calculate_strategy_payoff(self, 
                                 strategy: StrategyDefinition,
                                 underlying_prices: np.ndarray,
                                 volatility: float) -> Dict[str, Any]:
        """
        Calculate strategy payoff across range of underlying prices.
        
        Args:
            strategy: Strategy definition
            underlying_prices: Array of underlying prices to test
            volatility: Implied volatility for pricing
            
        Returns:
            Dictionary with payoff analysis
        """
        payoffs = np.zeros(len(underlying_prices))
        option_values = np.zeros(len(underlying_prices))
        
        for i, S in enumerate(underlying_prices):
            total_payoff = 0.0
            total_option_value = 0.0
            
            for leg in strategy.legs:
                # Intrinsic value at expiration
                if leg.option_type == OptionType.CALL:
                    intrinsic_value = max(S - leg.strike_price, 0)
                else:
                    intrinsic_value = max(leg.strike_price - S, 0)
                
                # Current option value (for P&L calculation)
                option_price = self.calculator.black_scholes_price(
                    S, leg.strike_price, leg.time_to_expiry,
                    self.default_risk_free_rate, volatility, leg.option_type
                )
                
                total_payoff += leg.position * leg.quantity * intrinsic_value
                total_option_value += leg.position * leg.quantity * option_price
            
            payoffs[i] = total_payoff
            option_values[i] = total_option_value
        
        # Calculate net P&L (payoff - initial cost)
        current_price = self.market_data['close_price'].iloc[-1] if 'close_price' in self.market_data.columns else 100
        initial_cost = self._calculate_initial_cost(strategy, current_price, volatility)
        net_pnl = payoffs - initial_cost
        
        # Find breakeven points
        breakeven_points = self._find_breakeven_points(underlying_prices, net_pnl)
        
        return {
            'underlying_prices': underlying_prices,
            'gross_payoffs': payoffs,
            'option_values': option_values,
            'net_pnl': net_pnl,
            'initial_cost': initial_cost,
            'max_profit': np.max(net_pnl),
            'max_loss': np.min(net_pnl),
            'breakeven_points': breakeven_points
        }
    
    def _calculate_initial_cost(self, 
                               strategy: StrategyDefinition,
                               current_price: float,
                               volatility: float) -> float:
        """Calculate initial cost/credit of the strategy."""
        
        total_cost = 0.0
        
        for leg in strategy.legs:
            option_price = self.calculator.black_scholes_price(
                current_price, leg.strike_price, leg.time_to_expiry,
                self.default_risk_free_rate, volatility, leg.option_type
            )
            
            # Positive for long positions (cost), negative for short positions (credit)
            total_cost += leg.position * leg.quantity * option_price
        
        return total_cost
    
    def _find_breakeven_points(self, 
                              prices: np.ndarray, 
                              pnl: np.ndarray) -> List[float]:
        """Find breakeven points where P&L crosses zero."""
        
        breakeven_points = []
        
        for i in range(len(pnl) - 1):
            if pnl[i] * pnl[i + 1] < 0:  # Sign change indicates breakeven
                # Linear interpolation to find exact breakeven point
                breakeven = prices[i] - pnl[i] * (prices[i + 1] - prices[i]) / (pnl[i + 1] - pnl[i])
                breakeven_points.append(breakeven)
        
        return breakeven_points
    
    def optimize_strategy_parameters(self, 
                                    strategy_type: StrategyType,
                                    target_month: int,
                                    optimization_criteria: str = "sharpe") -> StrategyDefinition:
        """
        Optimize strategy parameters using historical data.
        
        Args:
            strategy_type: Type of strategy to optimize
            target_month: Target month for the strategy
            optimization_criteria: Criteria for optimization (sharpe, profit, etc.)
            
        Returns:
            Optimized strategy definition
        """
        
        if target_month not in self.seasonality:
            raise ValueError(f"No seasonality data for month {target_month}")
        
        month_stats = self.seasonality[target_month]
        volatility = month_stats.std_return
        
        # Define parameter ranges for optimization
        if strategy_type == StrategyType.PUT_SPREAD:
            # Optimize strike selection
            current_price = self.market_data['close_price'].iloc[-1] if 'close_price' in self.market_data.columns else 100
            
            best_strategy = None
            best_score = -np.inf
            
            # Grid search over strike ratios
            long_strike_ratios = np.arange(0.85, 0.95, 0.02)
            short_strike_ratios = np.arange(0.90, 1.00, 0.02)
            
            for long_ratio in long_strike_ratios:
                for short_ratio in short_strike_ratios:
                    if long_ratio >= short_ratio:
                        continue  # Invalid spread
                    
                    strategy = StrategyDefinition(
                        strategy_type=StrategyType.PUT_SPREAD,
                        legs=[
                            StrategyLeg(OptionType.PUT, current_price * long_ratio, 
                                      self.days_to_expiry / 365.25, 1, 1),
                            StrategyLeg(OptionType.PUT, current_price * short_ratio, 
                                      self.days_to_expiry / 365.25, -1, 1)
                        ],
                        target_month=target_month,
                        entry_days_before=5,
                        exit_days_before=5
                    )
                    
                    # Evaluate strategy
                    score = self._evaluate_strategy_score(strategy, optimization_criteria)
                    
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy
            
            return best_strategy
        
        else:
            # For other strategies, use default design
            if strategy_type == StrategyType.CALL_SPREAD:
                return self.design_call_spread_strategy(target_month)
            elif strategy_type == StrategyType.STRADDLE:
                return self.design_straddle_strategy(target_month)
            else:
                raise NotImplementedError(f"Optimization not implemented for {strategy_type}")
    
    def _evaluate_strategy_score(self, 
                                strategy: StrategyDefinition,
                                criteria: str) -> float:
        """Evaluate strategy using specified criteria."""
        
        # Simple scoring based on expected returns and risk
        # In practice, this would use historical backtesting
        
        current_price = self.market_data['close_price'].iloc[-1] if 'close_price' in self.market_data.columns else 100
        month_stats = self.seasonality[strategy.target_month]
        
        # Calculate expected payoff
        price_range = np.linspace(current_price * 0.8, current_price * 1.2, 100)
        payoff_analysis = self.calculate_strategy_payoff(strategy, price_range, month_stats.std_return)
        
        expected_return = np.mean(payoff_analysis['net_pnl'])
        risk = np.std(payoff_analysis['net_pnl'])
        
        if criteria == "sharpe":
            return expected_return / max(risk, 0.001)  # Avoid division by zero
        elif criteria == "profit":
            return expected_return
        elif criteria == "max_profit":
            return payoff_analysis['max_profit']
        else:
            return expected_return / max(abs(payoff_analysis['max_loss']), 0.001)


class StrategyBacktester:
    """Backtest seasonal options strategies."""
    
    def __init__(self, market_data: pd.DataFrame):
        """Initialize backtester with market data."""
        self.market_data = market_data
        self.calculator = OptionsCalculator()
        self.logger = get_logger(__name__)
    
    def backtest_strategy(self, 
                         strategy: StrategyDefinition,
                         start_date: datetime,
                         end_date: datetime,
                         volatility_model: str = "historical") -> Dict[str, Any]:
        """
        Backtest a strategy over a date range.
        
        Args:
            strategy: Strategy definition to backtest
            start_date: Start date for backtesting
            end_date: End date for backtesting
            volatility_model: Method for estimating volatility
            
        Returns:
            Backtesting results
        """
        
        # Filter data for backtest period
        backtest_data = self.market_data[
            (self.market_data.index >= start_date) & 
            (self.market_data.index <= end_date)
        ].copy()
        
        if backtest_data.empty:
            return {'error': 'No data in backtest period'}
        
        # Calculate returns
        if 'returns' not in backtest_data.columns:
            backtest_data['returns'] = backtest_data['close_price'].pct_change()
        
        # Find all instances of target month
        target_month_instances = self._find_strategy_entry_dates(
            backtest_data, strategy.target_month, strategy.entry_days_before
        )
        
        trades = []
        
        for entry_date, exit_date in target_month_instances:
            trade_result = self._simulate_trade(
                strategy, entry_date, exit_date, backtest_data, volatility_model
            )
            if trade_result:
                trades.append(trade_result)
        
        if not trades:
            return {'error': 'No valid trades found in backtest period'}
        
        # Aggregate results
        results = self._aggregate_backtest_results(trades)
        results['individual_trades'] = trades
        results['strategy'] = strategy
        results['backtest_period'] = {'start': start_date, 'end': end_date}
        
        return results
    
    def _find_strategy_entry_dates(self, 
                                  data: pd.DataFrame,
                                  target_month: int,
                                  entry_days_before: int) -> List[Tuple[datetime, datetime]]:
        """Find entry and exit dates for the strategy."""
        
        entry_exit_pairs = []
        
        # Group by year to find each occurrence of target month
        for year in data.index.year.unique():
            year_data = data[data.index.year == year]
            
            # Find first day of target month
            target_month_data = year_data[year_data.index.month == target_month]
            
            if not target_month_data.empty:
                first_day_of_month = target_month_data.index.min()
                
                # Calculate entry date (days before month start)
                entry_date = first_day_of_month - pd.Timedelta(days=entry_days_before)
                
                # Calculate exit date (assuming 30-day holding period)
                exit_date = entry_date + pd.Timedelta(days=30)
                
                # Ensure dates are in our data range
                if entry_date in data.index and exit_date in data.index:
                    entry_exit_pairs.append((entry_date, exit_date))
        
        return entry_exit_pairs
    
    def _simulate_trade(self, 
                       strategy: StrategyDefinition,
                       entry_date: datetime,
                       exit_date: datetime,
                       data: pd.DataFrame,
                       volatility_model: str) -> Optional[Dict[str, Any]]:
        """Simulate a single trade."""
        
        if entry_date not in data.index or exit_date not in data.index:
            return None
        
        entry_price = data.loc[entry_date, 'close_price']
        exit_price = data.loc[exit_date, 'close_price']
        
        # Estimate volatility
        if volatility_model == "historical":
            # Use 30-day historical volatility
            returns_window = data.loc[:entry_date, 'returns'].tail(30)
            volatility = returns_window.std() * np.sqrt(252)  # Annualize
        else:
            volatility = 0.2  # Default 20%
        
        # Calculate entry cost
        entry_cost = 0.0
        exit_value = 0.0
        
        for leg in strategy.legs:
            # Entry option price
            entry_option_price = self.calculator.black_scholes_price(
                entry_price, leg.strike_price, leg.time_to_expiry,
                JapaneseMarketConstants.DEFAULT_RISK_FREE_RATE, volatility, leg.option_type
            )
            
            # Exit option value (intrinsic value at expiry)
            if leg.option_type == OptionType.CALL:
                exit_option_value = max(exit_price - leg.strike_price, 0)
            else:
                exit_option_value = max(leg.strike_price - exit_price, 0)
            
            entry_cost += leg.position * leg.quantity * entry_option_price
            exit_value += leg.position * leg.quantity * exit_option_value
        
        pnl = exit_value - entry_cost
        return_pct = pnl / max(abs(entry_cost), 1)  # Avoid division by zero
        
        return {
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_cost': entry_cost,
            'exit_value': exit_value,
            'pnl': pnl,
            'return_pct': return_pct,
            'volatility_used': volatility
        }
    
    def _aggregate_backtest_results(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate individual trade results."""
        
        trade_returns = [trade['return_pct'] for trade in trades]
        trade_pnls = [trade['pnl'] for trade in trades]
        
        total_return = sum(trade_returns)
        avg_return = np.mean(trade_returns)
        volatility = np.std(trade_returns)
        
        # Performance metrics
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
        
        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe_ratio = avg_return / max(volatility, 0.001)
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(trade_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'total_trades': len(trades),
            'total_return': total_return,
            'average_return': avg_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'best_trade': max(trade_returns),
            'worst_trade': min(trade_returns),
            'total_pnl': sum(trade_pnls),
            'average_pnl': np.mean(trade_pnls)
        }