#!/usr/bin/env python3
"""
Example usage script for Nikkei 225 Seasonality Analysis.

This script demonstrates how to use the various components of the analysis
system programmatically.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src import (
    AnalysisPipeline, SystemConfig, SeasonalityAnalyzer, 
    OptionsCalculator, MonteCarloEngine, setup_logging
)
from src.options import StrategyType
from src.risk import ProcessParameters, StochasticProcess


async def example_full_pipeline():
    """Demonstrate complete analysis pipeline."""
    
    print("="*60)
    print("EXAMPLE 1: COMPLETE ANALYSIS PIPELINE")
    print("="*60)
    
    # Initialize configuration and pipeline
    config = SystemConfig()
    pipeline = AnalysisPipeline(config)
    
    # Define analysis period (last 3 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    
    print(f"Running analysis from {start_date.date()} to {end_date.date()}")
    
    try:
        # Run complete analysis
        results = await pipeline.run_full_analysis(
            start_date=start_date,
            end_date=end_date,
            save_results=True
        )
        
        if results['success']:
            print("✓ Analysis completed successfully!")
            
            # Print key findings
            if 'analysis_phase' in results:
                significant_months = results['analysis_phase'].get('significant_months', [])
                print(f"Found {len(significant_months)} months with significant seasonal patterns")
                
                if significant_months:
                    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    print(f"Significant months: {[month_names[m] for m in significant_months]}")
            
            # Print strategy results
            if 'strategy_phase' in results:
                strategies_developed = results['strategy_phase']['total_strategies_developed']
                print(f"Developed {strategies_developed} option strategies")
        
        else:
            print(f"✗ Analysis failed: {results.get('error')}")
    
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")


def example_seasonality_analysis():
    """Demonstrate standalone seasonality analysis."""
    
    print("\n" + "="*60)
    print("EXAMPLE 2: SEASONALITY ANALYSIS WITH SAMPLE DATA")
    print("="*60)
    
    # Generate sample Nikkei 225-like data
    np.random.seed(42)  # For reproducibility
    
    # Create 5 years of daily data
    dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='D')
    dates = dates[dates.dayofweek < 5]  # Remove weekends
    
    # Simulate price data with some seasonal patterns
    n_days = len(dates)
    
    # Base returns with slight positive drift
    base_returns = np.random.normal(0.0003, 0.015, n_days)
    
    # Add seasonal effects (March negative, May positive as examples)
    seasonal_effects = np.zeros(n_days)
    for i, date in enumerate(dates):
        if date.month == 3:  # March effect (negative)
            seasonal_effects[i] = -0.005
        elif date.month == 5:  # May effect (positive)
            seasonal_effects[i] = 0.004
        elif date.month == 11:  # November effect (positive)
            seasonal_effects[i] = 0.003
    
    returns = base_returns + seasonal_effects
    
    # Generate price series
    initial_price = 25000  # Typical Nikkei level
    prices = [initial_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create DataFrame
    data = pd.DataFrame({
        'close_price': prices[1:],  # Remove initial price
        'returns': returns
    }, index=dates)
    
    data['adjusted_close'] = data['close_price']  # For simplicity
    
    print(f"Generated {len(data)} days of sample data")
    print(f"Price range: {data['close_price'].min():.0f} - {data['close_price'].max():.0f}")
    
    # Run seasonality analysis
    analyzer = SeasonalityAnalyzer(data, significance_level=0.05)
    
    # Test monthly patterns
    monthly_results = analyzer.test_monthly_patterns()
    
    print("\nMonthly Seasonality Results:")
    print("-" * 50)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    significant_months = []
    
    for month, result in monthly_results.items():
        status = "✓ SIGNIFICANT" if result.is_significant else "  Not significant"
        mean_ret = result.mean_return * 100
        t_stat = result.t_statistic
        p_val = result.t_pvalue
        
        print(f"{month_names[month-1]}: {mean_ret:+6.2f}% (t={t_stat:6.2f}, p={p_val:.3f}) {status}")
        
        if result.is_significant:
            significant_months.append(month_names[month-1])
    
    print(f"\nFound significant patterns in: {', '.join(significant_months) if significant_months else 'No months'}")
    
    # Test day-of-week patterns
    dow_results = analyzer.test_day_of_week_patterns()
    
    print("\nDay-of-Week Effects:")
    print("-" * 30)
    
    for dow, result in dow_results.items():
        day_name = result['day_name']
        mean_ret = result['mean_return'] * 100
        significant = "✓" if result['is_significant'] else " "
        print(f"{significant} {day_name}: {mean_ret:+6.2f}%")
    
    return data, monthly_results


def example_options_strategy():
    """Demonstrate options strategy development."""
    
    print("\n" + "="*60)
    print("EXAMPLE 3: OPTIONS STRATEGY DEVELOPMENT")
    print("="*60)
    
    # Use sample data from previous example
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    dates = dates[dates.dayofweek < 5]
    
    # Generate data with March seasonal effect
    returns = np.random.normal(0.0003, 0.015, len(dates))
    for i, date in enumerate(dates):
        if date.month == 3:
            returns[i] += -0.008  # Strong March effect
    
    prices = [25000]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'close_price': prices[1:],
        'returns': returns
    }, index=dates)
    
    # Create mock seasonality results
    from src.analysis.seasonality import SeasonalityResult
    
    seasonality_results = {
        3: SeasonalityResult(  # March
            month=3,
            mean_return=-0.025,  # -2.5% average return
            std_return=0.045,    # 4.5% volatility
            skewness=-0.3,
            kurtosis=0.5,
            t_statistic=-2.8,
            t_pvalue=0.008,      # Significant
            is_significant=True,
            normality_test_statistic=1.2,
            normality_pvalue=0.3,
            is_normal=True,
            sample_size=60,
            confidence_interval=(-0.040, -0.010)
        )
    }
    
    # Initialize strategy engine
    from src.options import SeasonalOptionsStrategy
    strategy_engine = SeasonalOptionsStrategy(data, seasonality_results)
    
    # Design put spread for March (expecting decline)
    strategy = strategy_engine.design_put_spread_strategy(target_month=3)
    
    print("Put Spread Strategy for March:")
    print("-" * 40)
    print(f"Strategy Type: {strategy.strategy_type.value}")
    print(f"Target Month: {strategy.target_month}")
    print(f"Number of Legs: {len(strategy.legs)}")
    
    current_price = data['close_price'].iloc[-1]
    print(f"Current Price: {current_price:.0f}")
    
    for i, leg in enumerate(strategy.legs):
        position_type = "LONG" if leg.position > 0 else "SHORT"
        print(f"Leg {i+1}: {position_type} {leg.option_type.value.upper()} @ Strike {leg.strike_price:.0f}")
    
    # Calculate strategy payoff
    price_range = np.linspace(current_price * 0.85, current_price * 1.15, 100)
    payoff_analysis = strategy_engine.calculate_strategy_payoff(
        strategy, price_range, seasonality_results[3].std_return
    )
    
    print(f"\nStrategy Analysis:")
    print(f"Initial Cost: {payoff_analysis['initial_cost']:.0f}")
    print(f"Max Profit: {payoff_analysis['max_profit']:.0f}")
    print(f"Max Loss: {payoff_analysis['max_loss']:.0f}")
    print(f"Breakeven Points: {[f'{be:.0f}' for be in payoff_analysis['breakeven_points']]}")
    
    return strategy, payoff_analysis


def example_monte_carlo_risk():
    """Demonstrate Monte Carlo risk analysis."""
    
    print("\n" + "="*60)
    print("EXAMPLE 4: MONTE CARLO RISK ANALYSIS")
    print("="*60)
    
    # Initialize Monte Carlo engine
    mc_engine = MonteCarloEngine(num_simulations=5000)  # Reduced for example
    
    # Define process parameters (based on Nikkei characteristics)
    parameters = ProcessParameters(
        mu=0.05,     # 5% annual drift
        sigma=0.20   # 20% annual volatility
    )
    
    # Simulate price paths
    initial_price = 28000
    time_horizon = 30/365  # 30 days
    num_steps = 30
    
    print(f"Simulating {mc_engine.num_simulations} price paths")
    print(f"Initial Price: {initial_price}")
    print(f"Time Horizon: {time_horizon*365:.0f} days")
    
    price_paths = mc_engine.simulate_price_paths(
        initial_price=initial_price,
        process=StochasticProcess.GEOMETRIC_BROWNIAN_MOTION,
        parameters=parameters,
        time_horizon=time_horizon,
        num_steps=num_steps
    )
    
    # Analyze final prices
    final_prices = price_paths[:, -1]
    
    print(f"\nSimulation Results:")
    print(f"Mean Final Price: {np.mean(final_prices):.0f}")
    print(f"Std Final Price: {np.std(final_prices):.0f}")
    print(f"Min Final Price: {np.min(final_prices):.0f}")
    print(f"Max Final Price: {np.max(final_prices):.0f}")
    
    # Calculate price change statistics
    price_changes = (final_prices - initial_price) / initial_price
    
    print(f"\nPrice Change Statistics:")
    print(f"Mean Change: {np.mean(price_changes)*100:+.2f}%")
    print(f"Std Change: {np.std(price_changes)*100:.2f}%")
    print(f"95% VaR: {np.percentile(price_changes, 5)*100:.2f}%")
    print(f"99% VaR: {np.percentile(price_changes, 1)*100:.2f}%")
    
    # Probability of different outcomes
    prob_up_5 = np.mean(price_changes > 0.05)
    prob_down_5 = np.mean(price_changes < -0.05)
    
    print(f"\nProbabilities:")
    print(f"P(Price up >5%): {prob_up_5:.1%}")
    print(f"P(Price down >5%): {prob_down_5:.1%}")
    
    return price_paths, parameters


def example_options_pricing():
    """Demonstrate options pricing calculations."""
    
    print("\n" + "="*60)
    print("EXAMPLE 5: OPTIONS PRICING")
    print("="*60)
    
    # Initialize options calculator
    calc = OptionsCalculator()
    
    # Option parameters (typical for Nikkei options)
    S = 28000      # Current Nikkei level
    K = 28000      # ATM strike
    T = 30/365     # 30 days to expiry
    r = 0.001      # 0.1% risk-free rate (Japan)
    sigma = 0.20   # 20% implied volatility
    
    from src.options import OptionType
    
    # Calculate option prices
    call_price = calc.black_scholes_price(S, K, T, r, sigma, OptionType.CALL)
    put_price = calc.black_scholes_price(S, K, T, r, sigma, OptionType.PUT)
    
    print(f"Option Pricing (ATM, 30 days to expiry):")
    print(f"Underlying: {S}")
    print(f"Strike: {K}")
    print(f"Volatility: {sigma*100:.0f}%")
    print(f"Time to Expiry: {T*365:.0f} days")
    print(f"Risk-free Rate: {r*100:.1f}%")
    
    print(f"\nOption Prices:")
    print(f"Call Price: {call_price:.0f}")
    print(f"Put Price: {put_price:.0f}")
    print(f"Put-Call Parity Check: {call_price - put_price:.2f} vs {S - K*np.exp(-r*T):.2f}")
    
    # Calculate Greeks
    from src.options import GreeksCalculator
    greeks_calc = GreeksCalculator(calc)
    
    call_greeks = greeks_calc.calculate_greeks(S, K, T, r, sigma, OptionType.CALL)
    put_greeks = greeks_calc.calculate_greeks(S, K, T, r, sigma, OptionType.PUT)
    
    print(f"\nCall Option Greeks:")
    print(f"Delta: {call_greeks.delta:.3f}")
    print(f"Gamma: {call_greeks.gamma:.6f}")
    print(f"Theta: {call_greeks.theta:.2f} (per day)")
    print(f"Vega: {call_greeks.vega:.2f} (per 1% vol)")
    
    print(f"\nPut Option Greeks:")
    print(f"Delta: {put_greeks.delta:.3f}")
    print(f"Gamma: {put_greeks.gamma:.6f}")
    print(f"Theta: {put_greeks.theta:.2f} (per day)")
    print(f"Vega: {put_greeks.vega:.2f} (per 1% vol)")
    
    # Volatility surface (simplified)
    strikes = np.array([26000, 27000, 28000, 29000, 30000])
    vols = np.array([0.22, 0.21, 0.20, 0.21, 0.23])  # Typical smile
    
    print(f"\nVolatility Smile:")
    for strike, vol in zip(strikes, vols):
        moneyness = strike / S
        print(f"Strike {strike} (K/S={moneyness:.3f}): {vol*100:.0f}% vol")
    
    return call_price, put_price, call_greeks, put_greeks


async def run_all_examples():
    """Run all examples."""
    
    print("NIKKEI 225 SEASONALITY ANALYSIS - EXAMPLE USAGE")
    print("="*60)
    
    # Setup logging
    setup_logging()
    
    try:
        # Example 1: Full pipeline (commented out for demo as it takes time)
        # await example_full_pipeline()
        
        # Example 2: Seasonality analysis with sample data
        data, seasonality_results = example_seasonality_analysis()
        
        # Example 3: Options strategy development
        strategy, payoff_analysis = example_options_strategy()
        
        # Example 4: Monte Carlo risk analysis
        price_paths, parameters = example_monte_carlo_risk()
        
        # Example 5: Options pricing
        call_price, put_price, call_greeks, put_greeks = example_options_pricing()
        
        print(f"\n{'='*60}")
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nTo run the full analysis pipeline with real data, use:")
        print("python main.py full-analysis --years 5")
        
    except Exception as e:
        print(f"\nExample failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_examples())