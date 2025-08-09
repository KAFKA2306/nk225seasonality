"""
Options pricing and Greeks calculation module.

This module implements advanced options pricing models including Black-Scholes,
binomial trees, and Monte Carlo methods, along with comprehensive Greeks calculation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from scipy.stats import norm
from scipy.optimize import brentq
import warnings

from ..config import get_logger, JapaneseMarketConstants


class OptionType(Enum):
    """Option types."""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionContract:
    """Represents an option contract."""
    
    underlying_price: float
    strike_price: float
    time_to_expiry: float  # In years
    risk_free_rate: float
    volatility: float
    option_type: OptionType
    dividend_yield: float = 0.0


@dataclass
class GreeksResult:
    """Option Greeks calculation result."""
    
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    price: float


class OptionsCalculator:
    """Advanced options pricing and Greeks calculation."""
    
    def __init__(self):
        """Initialize the options calculator."""
        self.logger = get_logger(__name__)
        
        # Default parameters for Japanese market
        self.default_risk_free_rate = JapaneseMarketConstants.DEFAULT_RISK_FREE_RATE
        self.trading_days_per_year = 252
        
    def black_scholes_price(self, 
                           S: float, 
                           K: float, 
                           T: float, 
                           r: float, 
                           sigma: float, 
                           option_type: OptionType,
                           q: float = 0.0) -> float:
        """
        Calculate Black-Scholes option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Call or put
            q: Dividend yield
            
        Returns:
            Option price
        """
        if T <= 0:
            # Option at expiration
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        if sigma <= 0:
            # Zero volatility case
            if option_type == OptionType.CALL:
                return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
            else:
                return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
        
        try:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == OptionType.CALL:
                price = (S * np.exp(-q * T) * norm.cdf(d1) - 
                        K * np.exp(-r * T) * norm.cdf(d2))
            else:
                price = (K * np.exp(-r * T) * norm.cdf(-d2) - 
                        S * np.exp(-q * T) * norm.cdf(-d1))
            
            return max(price, 0.0)  # Ensure non-negative price
            
        except Exception as e:
            self.logger.error(f"Black-Scholes calculation error: {e}")
            return 0.0
    
    def binomial_tree_price(self, 
                           S: float, 
                           K: float, 
                           T: float, 
                           r: float, 
                           sigma: float, 
                           option_type: OptionType,
                           steps: int = 100,
                           american: bool = False) -> float:
        """
        Calculate option price using binomial tree method.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Call or put
            steps: Number of time steps
            american: Whether option is American style
            
        Returns:
            Option price
        """
        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        # Initialize asset prices at maturity
        stock_prices = np.zeros(steps + 1)
        for i in range(steps + 1):
            stock_prices[i] = S * (u ** (steps - i)) * (d ** i)
        
        # Initialize option values at maturity
        option_values = np.zeros(steps + 1)
        for i in range(steps + 1):
            if option_type == OptionType.CALL:
                option_values[i] = max(stock_prices[i] - K, 0)
            else:
                option_values[i] = max(K - stock_prices[i], 0)
        
        # Backward induction
        for step in range(steps - 1, -1, -1):
            for i in range(step + 1):
                # Calculate continuation value
                continuation_value = np.exp(-r * dt) * (
                    p * option_values[i] + (1 - p) * option_values[i + 1]
                )
                
                if american:
                    # Calculate intrinsic value
                    current_stock_price = S * (u ** (step - i)) * (d ** i)
                    if option_type == OptionType.CALL:
                        intrinsic_value = max(current_stock_price - K, 0)
                    else:
                        intrinsic_value = max(K - current_stock_price, 0)
                    
                    option_values[i] = max(continuation_value, intrinsic_value)
                else:
                    option_values[i] = continuation_value
        
        return option_values[0]
    
    def monte_carlo_price(self, 
                         S: float, 
                         K: float, 
                         T: float, 
                         r: float, 
                         sigma: float, 
                         option_type: OptionType,
                         simulations: int = 100000,
                         antithetic: bool = True) -> Tuple[float, float]:
        """
        Calculate option price using Monte Carlo simulation.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Call or put
            simulations: Number of Monte Carlo simulations
            antithetic: Use antithetic variates for variance reduction
            
        Returns:
            Tuple of (option_price, standard_error)
        """
        np.random.seed(42)  # For reproducibility
        
        if antithetic:
            # Generate half the number of random variables
            z = np.random.standard_normal(simulations // 2)
            # Create antithetic pairs
            z = np.concatenate([z, -z])
        else:
            z = np.random.standard_normal(simulations)
        
        # Generate final stock prices
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        
        # Calculate payoffs
        if option_type == OptionType.CALL:
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Discount to present value
        price = np.exp(-r * T) * np.mean(payoffs)
        standard_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(simulations)
        
        return price, standard_error
    
    def implied_volatility(self, 
                          market_price: float,
                          S: float, 
                          K: float, 
                          T: float, 
                          r: float, 
                          option_type: OptionType,
                          q: float = 0.0,
                          max_iterations: int = 100,
                          tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Brent's method.
        
        Args:
            market_price: Observed market price
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            option_type: Call or put
            q: Dividend yield
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility
        """
        def price_diff(vol):
            try:
                bs_price = self.black_scholes_price(S, K, T, r, vol, option_type, q)
                return bs_price - market_price
            except:
                return float('inf')
        
        # Check boundary conditions
        if market_price <= 0:
            return 0.0
        
        # Set bounds for volatility search
        vol_low = 0.001  # 0.1%
        vol_high = 5.0   # 500%
        
        # Check if solution exists within bounds
        if price_diff(vol_low) * price_diff(vol_high) > 0:
            self.logger.warning("Implied volatility may not exist within bounds")
            return np.nan
        
        try:
            implied_vol = brentq(price_diff, vol_low, vol_high, 
                                maxiter=max_iterations, xtol=tolerance)
            return implied_vol
        except ValueError as e:
            self.logger.error(f"Implied volatility calculation failed: {e}")
            return np.nan


class GreeksCalculator:
    """Calculate option Greeks."""
    
    def __init__(self, calculator: OptionsCalculator):
        """Initialize Greeks calculator."""
        self.calculator = calculator
        self.logger = get_logger(__name__)
    
    def calculate_greeks(self, 
                        S: float, 
                        K: float, 
                        T: float, 
                        r: float, 
                        sigma: float, 
                        option_type: OptionType,
                        q: float = 0.0) -> GreeksResult:
        """
        Calculate all Greeks for an option.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Call or put
            q: Dividend yield
            
        Returns:
            GreeksResult with all Greeks
        """
        if T <= 0:
            return GreeksResult(
                delta=0.0, gamma=0.0, theta=0.0, 
                vega=0.0, rho=0.0, 
                price=max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0)
            )
        
        try:
            # Calculate d1 and d2
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Price
            price = self.calculator.black_scholes_price(S, K, T, r, sigma, option_type, q)
            
            # Delta
            if option_type == OptionType.CALL:
                delta = np.exp(-q * T) * norm.cdf(d1)
            else:
                delta = -np.exp(-q * T) * norm.cdf(-d1)
            
            # Gamma (same for calls and puts)
            gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
            
            # Theta
            if option_type == OptionType.CALL:
                theta = ((-S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) -
                        r * K * np.exp(-r * T) * norm.cdf(d2) +
                        q * S * np.exp(-q * T) * norm.cdf(d1)) / 365.25  # Convert to daily
            else:
                theta = ((-S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) +
                        r * K * np.exp(-r * T) * norm.cdf(-d2) -
                        q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365.25  # Convert to daily
            
            # Vega (same for calls and puts)
            vega = (S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)) / 100  # Convert to 1% vol change
            
            # Rho
            if option_type == OptionType.CALL:
                rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100  # Convert to 1% rate change
            else:
                rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100  # Convert to 1% rate change
            
            return GreeksResult(
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                price=price
            )
            
        except Exception as e:
            self.logger.error(f"Greeks calculation error: {e}")
            return GreeksResult(
                delta=0.0, gamma=0.0, theta=0.0, 
                vega=0.0, rho=0.0, price=0.0
            )
    
    def calculate_portfolio_greeks(self, 
                                  positions: List[Dict[str, Any]]) -> GreeksResult:
        """
        Calculate portfolio-level Greeks.
        
        Args:
            positions: List of position dictionaries with contract details and quantities
            
        Returns:
            Portfolio Greeks
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0
        total_value = 0.0
        
        for position in positions:
            contract = position['contract']
            quantity = position['quantity']
            
            greeks = self.calculate_greeks(
                contract['S'], contract['K'], contract['T'],
                contract['r'], contract['sigma'], contract['option_type'],
                contract.get('q', 0.0)
            )
            
            total_delta += quantity * greeks.delta
            total_gamma += quantity * greeks.gamma
            total_theta += quantity * greeks.theta
            total_vega += quantity * greeks.vega
            total_rho += quantity * greeks.rho
            total_value += quantity * greeks.price
        
        return GreeksResult(
            delta=total_delta,
            gamma=total_gamma,
            theta=total_theta,
            vega=total_vega,
            rho=total_rho,
            price=total_value
        )
    
    def delta_hedge_ratio(self, 
                         option_delta: float,
                         underlying_price: float) -> float:
        """
        Calculate delta hedge ratio.
        
        Args:
            option_delta: Option delta
            underlying_price: Current underlying price
            
        Returns:
            Number of shares to hedge per option contract
        """
        return -option_delta  # Negative because we're hedging
    
    def calculate_Greeks_surface(self, 
                               spot_range: np.ndarray,
                               vol_range: np.ndarray,
                               K: float,
                               T: float,
                               r: float,
                               option_type: OptionType) -> Dict[str, np.ndarray]:
        """
        Calculate Greeks surface across spot prices and volatilities.
        
        Args:
            spot_range: Array of spot prices
            vol_range: Array of volatilities
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            option_type: Call or put
            
        Returns:
            Dictionary of Greeks surfaces
        """
        spot_grid, vol_grid = np.meshgrid(spot_range, vol_range)
        
        delta_surface = np.zeros_like(spot_grid)
        gamma_surface = np.zeros_like(spot_grid)
        theta_surface = np.zeros_like(spot_grid)
        vega_surface = np.zeros_like(spot_grid)
        price_surface = np.zeros_like(spot_grid)
        
        for i in range(len(vol_range)):
            for j in range(len(spot_range)):
                greeks = self.calculate_greeks(
                    spot_grid[i, j], K, T, r, vol_grid[i, j], option_type
                )
                
                delta_surface[i, j] = greeks.delta
                gamma_surface[i, j] = greeks.gamma
                theta_surface[i, j] = greeks.theta
                vega_surface[i, j] = greeks.vega
                price_surface[i, j] = greeks.price
        
        return {
            'delta': delta_surface,
            'gamma': gamma_surface,
            'theta': theta_surface,
            'vega': vega_surface,
            'price': price_surface,
            'spot_grid': spot_grid,
            'vol_grid': vol_grid
        }