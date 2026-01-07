from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from ..config import JapaneseMarketConstants


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
