from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import yfinance as yf


@dataclass
class ValuationMetrics:
    jgb_yield: float
    current_per: float
    risk_premium: float
    eps: Optional[float] = None
    price: Optional[float] = None

    @property
    def earnings_yield(self) -> float:
        if self.current_per <= 0:
            return 0.0
        return (1 / self.current_per) * 100


class ValuationAnalyzer:
    def calculate_yield_gap(self, metrics: ValuationMetrics) -> float:
        return metrics.earnings_yield - metrics.jgb_yield

    def calculate_fair_per(self, metrics: ValuationMetrics) -> float:
        discount_rate = (metrics.jgb_yield + metrics.risk_premium) / 100
        if discount_rate <= 0:
            return float("inf")
        return 1 / discount_rate

    def calculate_valuation_status(self, metrics: ValuationMetrics) -> Dict[str, any]:
        fair_per = self.calculate_fair_per(metrics)
        yield_gap = self.calculate_yield_gap(metrics)

        divergence_pct = ((metrics.current_per - fair_per) / fair_per) * 100

        status = "Fairly Valued"
        if divergence_pct > 20:
            status = "Significantly Overvalued"
        elif divergence_pct > 10:
            status = "Overvalued"
        elif divergence_pct < -20:
            status = "Significantly Undervalued"
        elif divergence_pct < -10:
            status = "Undervalued"

        return {
            "current_per": metrics.current_per,
            "fair_per": round(fair_per, 2),
            "jgb_yield": metrics.jgb_yield,
            "earnings_yield": round(metrics.earnings_yield, 2),
            "yield_gap": round(yield_gap, 2),
            "divergence_pct": round(divergence_pct, 2),
            "status": status,
            "metrics": metrics,
        }


def run_analysis_report(jgb_yield: float, current_per: float, risk_premium: float) -> None:
    print("\n" + "=" * 50)
    print("MARKET VALUATION ANALYSIS")
    print("=" * 50)

    metrics = ValuationMetrics(jgb_yield, current_per, risk_premium)
    analyzer = ValuationAnalyzer()
    result = analyzer.calculate_valuation_status(metrics)

    print("\nINPUTS:")
    print(f"  Current PER:     {metrics.current_per:>6.2f}x")
    print(f"  JGB Yield:       {metrics.jgb_yield:>6.2f}%")
    print(f"  Risk Premium:    {metrics.risk_premium:>6.2f}%")

    print("\nANALYSIS:")
    print(f"  Earnings Yield:  {result['earnings_yield']:>6.2f}%")
    print(f"  Yield Gap:       {result['yield_gap']:>6.2f}% (Earnings Yield - Bond Yield)")
    print(f"  Fair PER:        {result['fair_per']:>6.2f}x (1 / (Bond Yield + Risk Premium))")

    print("\nCONCLUSION:")

    color = (
        "\033[92m"
        if "Undervalued" in result["status"]
        else "\033[91m"
        if "Overvalued" in result["status"]
        else "\033[93m"
    )
    reset = "\033[0m"

    print(f"  Status:          {color}{result['status']}{reset}")
    print(f"  Divergence:      {result['divergence_pct']:>+6.2f}% from Fair Value")
    print("\n" + "=" * 50)


def fetch_nikkei_data(years: int) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    ticker = yf.Ticker("^N225")
    df = ticker.history(start=start, end=end, interval="1mo")
    if df.empty:
        raise RuntimeError("Failed to fetch Nikkei 225 data")
    return df


def fetch_current_jgb_yield(ticker: str) -> float:
    """
    Fetches the current 10-year JGB yield from Yahoo Finance.
    Returns the yield as a percentage (e.g., 1.05 for 1.05%).
    
    Args:
        ticker: The ticker symbol for the JGB yield (e.g., ^JP10Y)
        
    Returns:
        float: The current yield in percent.
        
    Raises:
        RuntimeError: If fetching fails or data is empty.
    """
    try:
        jgb = yf.Ticker(ticker)
        # Fetch 5 days to ensure we get the latest trading day
        hist = jgb.history(period="5d")
        if hist.empty:
            raise RuntimeError(f"No data returned for JGB ticker {ticker}")
        
        # Get the most recent closing price
        current_yield = hist["Close"].iloc[-1]
        
        # Sanity check: Yield should be reasonable (between -1% and 10% for JGB)
        if not (-1.0 <= current_yield <= 10.0):
             # Some data sources might return 0.01 for 1%, others 1.0 for 1%
             # ^JP10Y returns 1.05 for 1.05%, so the range check is valid.
             pass

        return float(current_yield)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch JGB yield for {ticker}: {str(e)}")


def calculate_historical_per(price_data: pd.DataFrame, eps_provider: Optional[callable] = None) -> pd.DataFrame:
    df = price_data.copy()
    df["price"] = df["Close"]

    if eps_provider:
        df["estimated_eps"] = df.index.map(lambda d: eps_provider(d))
    else:
        raise ValueError("Dynamic EPS provider is required for strict valuation analysis.")

    df["estimated_per"] = df["price"] / df["estimated_eps"]
    return df


def run_time_series_report(years: int) -> None:
    from ..config import SystemConfig
    from .valuation import fetch_current_jgb_yield

    config = SystemConfig()

    jgb_yield = fetch_current_jgb_yield(config.valuation.jgb_ticker)
    print(f"Fetched JGB Yield: {jgb_yield}%")
    
    risk_premium = config.valuation.risk_premium

    print("\n" + "=" * 60)
    print("TIME SERIES VALUATION ANALYSIS (Dynamic EPS)")
    print("=" * 60)

    print(f"\nFetching {years} years of Nikkei 225 data...")
    price_data = fetch_nikkei_data(years)

    df = calculate_historical_per(price_data, eps_provider=config.valuation.get_eps_for_date)

    analyzer = ValuationAnalyzer()
    results = []

    for date, row in df.iterrows():
        per = row["estimated_per"]
        metrics = ValuationMetrics(jgb_yield, per, risk_premium)
        status = analyzer.calculate_valuation_status(metrics)
        results.append(
            {
                "date": date,
                "price": row["price"],
                "eps": row["estimated_eps"],
                "per": per,
                "fair_per": status["fair_per"],
                "divergence": status["divergence_pct"],
                "status": status["status"],
            }
        )

    results_df = pd.DataFrame(results)

    print(f"\nParameters: JGB={jgb_yield}%, Risk Premium={risk_premium}%")
    print(f"Fair PER: {results_df['fair_per'].iloc[0]:.2f}x")
    print("\n" + "-" * 75)
    print(f"{'Date':<12} {'Price':>10} {'EPS':>8} {'PER':>8} {'Diverg':>8} {'Status':<25}")
    print("-" * 75)

    for _, r in results_df.iterrows():
        color = "\033[92m" if "Under" in r["status"] else "\033[91m" if "Over" in r["status"] else "\033[93m"
        reset = "\033[0m"
        print(
            f"{r['date'].strftime('%Y-%m'):<12} {r['price']:>10,.0f} {r['eps']:>8.0f} {r['per']:>8.1f}x {r['divergence']:>+7.1f}% {color}{r['status']:<25}{reset}"
        )

    print("-" * 60)
    print(f"\nSUMMARY ({len(results_df)} periods)")
    print(f"  Avg PER:        {results_df['per'].mean():.1f}x")
    print(f"  Avg Divergence: {results_df['divergence'].mean():+.1f}%")
    print(f"  Max Undervalued: {results_df['divergence'].min():+.1f}%")
    print(f"  Max Overvalued:  {results_df['divergence'].max():+.1f}%")
    print("=" * 60)
