# Nikkei 225 Seasonality Analysis Project Overview

## Project Purpose
Quantitative finance research project focused on analyzing Nikkei 225 seasonality patterns and developing option strategies based on seasonal effects in the Japanese stock market. The project leverages statistical analysis, data processing, and options pricing models to identify and exploit predictable seasonal patterns.

## Key Features
- Statistical analysis of Japanese stock market seasonality patterns
- Nikkei 225 historical data analysis and pattern recognition  
- Development of quantitative option strategies (put spreads, vertical spreads, far calls)
- Risk management and Monte Carlo simulations for strategy backtesting
- Implementation of rigorous statistical testing frameworks

## Specialized Agent
The repository includes a specialized `nikkei-seasonality-analyst` agent located at `.claude/agents/nikkei-seasonality-analyst.md` that should be used proactively for:
- Market analysis and statistical testing
- Options strategy development
- Japanese market institutional knowledge
- Financial library expertise (pandas, numpy, scipy, statsmodels, quantlib)

## Current Project Status
- Complete modular codebase structure implemented
- Core modules: data ingestion, analysis, options pricing, risk management, visualization
- Database integration with SQLite (nikkei_data.db)
- Comprehensive logging system
- Full test coverage framework ready
- Development tools configured (black, flake8, mypy)

## Data Sources
Designed to work with historical Nikkei 225 data from:
- Nikkei official data
- Investing.com
- JPX (Japan Exchange Group)
- Optional: Yahoo Finance, Alpha Vantage, Quandl

## Japanese Market Specifics
- Fiscal year-end effects (March 31st)
- Trading holidays and calendar effects
- Institutional behaviors and flows
- Index rebalancing effects