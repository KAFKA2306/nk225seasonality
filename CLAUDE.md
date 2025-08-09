# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative finance research project focused on analyzing Nikkei 225 seasonality patterns and developing option strategies based on seasonal effects in the Japanese stock market. The project leverages statistical analysis, data processing, and options pricing models to identify and exploit predictable seasonal patterns.

## Specialized Agent

This repository includes a specialized `nikkei-seasonality-analyst` agent located at `.claude/agents/nikkei-seasonality-analyst.md`. This agent should be used proactively for:

- Statistical analysis of Japanese stock market seasonality patterns
- Nikkei 225 historical data analysis and pattern recognition
- Development of quantitative option strategies (put spreads, vertical spreads, far calls)
- Risk management and Monte Carlo simulations for strategy backtesting
- Implementation of rigorous statistical testing frameworks

The agent has expertise in Python financial libraries (pandas, numpy, scipy, statsmodels, quantlib) and Japanese market institutional knowledge.

## Project Structure

- `specs/` - Contains project requirements and design documentation
- `data/scrape/` - Data collection and scraping functionality (directory structure prepared)
- `.claude/agents/` - Specialized AI agents for domain-specific tasks

## Development Approach

When working on this project:

1. Use the nikkei-seasonality-analyst agent for any market analysis, statistical testing, or options strategy development
2. Follow academic rigor in statistical analysis with proper significance testing and validation
3. Implement robust data validation and quality control for financial data
4. Account for Japanese market specifics: fiscal year-end effects, trading holidays, institutional behaviors
5. Maintain reproducible analysis pipelines with version control considerations

## Data Sources and Processing

The project is designed to work with historical Nikkei 225 data from multiple sources including Nikkei official data, Investing.com, and JPX. Data processing should handle missing data, trading holidays, and index rebalancing effects appropriately.