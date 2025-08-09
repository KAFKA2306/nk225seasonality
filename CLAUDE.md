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

## Common Development Commands

### Main Analysis Pipeline
```bash
# Run complete seasonality analysis
python main.py full-analysis --years 5

# Run analysis for specific date range  
python main.py full-analysis --start-date 2020-01-01 --end-date 2023-12-31

# Daily data update
python main.py daily-update

# Check system status
python main.py status

# Validate configuration
python main.py validate-config
```

### Testing and Code Quality
```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test modules
pytest tests/test_seasonality.py
pytest tests/test_options.py

# Format code
black src/ tests/

# Lint code  
flake8 src/ tests/

# Type checking
mypy src/
```

### Development and Analysis
```bash
# Run example usage script
python example_usage.py

# Start Jupyter notebooks
jupyter notebook

# Run individual modules
python -m src.analysis.seasonality
python -m src.options.strategies
python -m src.risk.monte_carlo
```

## Architecture Overview

The system follows a modular architecture with clear separation of concerns:

### Core Modules
- **`src/analysis/`** - Statistical analysis and seasonality detection
  - `seasonality.py` - Core seasonality testing (t-tests, ANOVA, regression)
  - `mechanism.py` - Factor attribution and causal analysis
- **`src/data/`** - Data ingestion, validation, and storage
  - `ingestion.py` - Multi-source data collection with failover
  - `validation.py` - Quality control with 15+ validation rules
  - `repository.py` - Database access using repository pattern
- **`src/options/`** - Options pricing and strategy development
  - `calculator.py` - Black-Scholes, binomial, Monte Carlo pricing
  - `strategies.py` - Seasonal strategy optimization
- **`src/risk/`** - Risk management and simulation
  - `monte_carlo.py` - Stochastic process simulation
  - `var_calculator.py` - VaR and Expected Shortfall calculations
- **`src/visualization/`** - Charts and reporting
  - Publication-quality seasonality heatmaps and risk dashboards
- **`src/config.py`** - System configuration and Japanese market constants

### Data Flow Pipeline
1. **Data Ingestion** → SQLite database (`data/nikkei_data.db`)
2. **Quality Validation** → Cleaned datasets with lineage tracking
3. **Statistical Analysis** → Seasonal pattern detection with significance testing
4. **Strategy Development** → Options strategies optimized for seasonal effects
5. **Risk Assessment** → Monte Carlo backtesting and VaR calculations
6. **Visualization** → Professional charts and analysis reports

### Key Design Patterns
- **Repository Pattern**: Centralized data access through `MarketDataRepository`
- **Strategy Pattern**: Pluggable options strategies via `SeasonalOptionsStrategy`
- **Pipeline Pattern**: Coordinated analysis workflow in `AnalysisPipeline`
- **Configuration Management**: Environment-specific settings via `SystemConfig`

## Development Workflow

### Code Quality Standards
- Follow TDD approach: write failing tests first
- All code must pass `black`, `flake8`, and `mypy` checks
- Maintain >90% test coverage
- Include comprehensive docstrings with academic references
- Implement proper error handling and logging

### Japanese Market Expertise Requirements
- Account for fiscal year-end effects (March 31 rebalancing)
- Handle Japanese trading holidays (Golden Week, Obon, year-end)
- Consider institutional behavior patterns and policy cycles
- Validate against known market anomalies and historical events
- Use JST timezone for all market data processing

### File Structure and Storage
- **Database**: SQLite at `data/nikkei_data.db`
- **Logs**: Structured logging in `logs/` directory  
- **Outputs**: Analysis results saved to `outputs/`
- **Notebooks**: Jupyter exploration notebooks in `notebooks/`
- **Configuration**: Use `.env` files for API keys and sensitive settings