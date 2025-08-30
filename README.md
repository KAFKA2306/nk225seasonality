# Nikkei 225 Seasonality Analysis System

A comprehensive quantitative finance platform for analyzing seasonal patterns in the Nikkei 225 index and developing data-driven options trading strategies.

## Overview

This system provides sophisticated statistical analysis of Japanese equity market seasonality, with specialized focus on:

- **Seasonal Pattern Detection**: Rigorous statistical testing of monthly, quarterly, and intraweek patterns
- **Options Strategy Development**: Black-Scholes pricing with seasonal optimization for put spreads, call spreads, and volatility strategies
- **Risk Management**: Monte Carlo simulation with comprehensive VaR and Expected Shortfall calculations
- **Japanese Market Expertise**: Built-in handling of fiscal year effects, trading holidays, and institutional factors

## Key Features

### 📊 Statistical Analysis
- **Seasonality Detection**: T-tests, ANOVA, and regression models with multiple comparison corrections
- **Mechanism Analysis**: Quantification of fiscal year-end effects, earnings seasons, and policy cycles
- **Pattern Validation**: Rolling window analysis and out-of-sample testing for robustness

### 💹 Options Pricing & Strategies
- **Advanced Pricing Models**: Black-Scholes, binomial trees, and Monte Carlo methods
- **Greeks Calculation**: Delta, gamma, theta, vega, and rho with numerical accuracy
- **Seasonal Strategies**: Optimized put spreads for March volatility, year-end rally strategies, summer volatility plays

### 🎯 Risk Management
- **Monte Carlo Engine**: Multiple stochastic processes (GBM, jump-diffusion, Heston)
- **VaR Calculations**: Parametric, historical, and extreme value theory methods
- **Stress Testing**: Scenario analysis with Japanese market-specific stress scenarios

### 📈 Data & Visualization
- **Multi-Source Integration**: Nikkei official, JPX, and Investing.com data feeds
- **Quality Validation**: 15+ validation rules with anomaly detection
- **Professional Charts**: Publication-quality seasonality heatmaps, payoff diagrams, risk dashboards

### Fundamental Metrics

Historical price-to-earnings (PER) and derived earnings-per-share (EPS) data for the Nikkei 225 over the last three years:

View live updates on the [GitHub Pages dashboard](https://<your-username>.github.io/nk225seasonality/).

#### PER (last 3 years)

```
   17.62  ┤
   17.10  ┤                                   ╭
   16.59  ┤                  ╭─╮╭╮            │
   16.07  ┤                 ╭╯ ╰╯╰╮          ╭╯
   15.56  ┤                ╭╯     ╰╮╭╮╭─╮  ╭─╯
   15.04  ┤         ╭───╮  │       ╰╯╰╯ │  │
   14.53  ┤         │   ╰──╯            ╰──╯
   14.01  ┤        ╭╯
   13.50  ┤       ╭╯
   12.98  ┼╮╭─╮╭──╯
   12.47  ┤││ ││
   11.95  ┤╰╯ ╰╯
```

#### EPS (last 3 years)

```
 2538.39  ┤
 2496.04  ┤                             ╭╮  ╭╮
 2453.68  ┤                        ╭────╯│  │╰╮
 2411.32  ┤                       ╭╯     ╰──╯ ╰
 2368.96  ┤                 ╭─╮ ╭─╯
 2326.60  ┤                 │ │╭╯
 2284.25  ┤               ╭─╯ ││
 2241.89  ┤              ╭╯   ╰╯
 2199.53  ┼─╮╭╮    ╭─╮   │
 2157.17  ┤ ╰╯╰─╮  │ ╰─╮ │
 2114.81  ┤     ╰──╯   ╰╮│
 2072.45  ┤             ╰╯
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Validate system configuration
python main.py validate-config
```

### Basic Usage

```bash
# Check system status
python main.py status

# Run complete seasonality analysis
python main.py full-analysis --years 5

# Run analysis for specific date range
python main.py full-analysis --start-date 2020-01-01 --end-date 2023-12-31

# Daily data update
python main.py daily-update
```

### Programmatic Usage

```python
from src import AnalysisPipeline, SystemConfig

# Initialize analysis pipeline
config = SystemConfig()
pipeline = AnalysisPipeline(config)

# Run comprehensive analysis
results = await pipeline.run_full_analysis(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31)
)

# Extract seasonal patterns
monthly_patterns = results['seasonality']['monthly_analysis']
significant_months = [month for month, data in monthly_patterns.items() 
                     if data['p_value'] < 0.05]
```

## Core Components

### Statistical Analysis Engine
- `SeasonalityAnalyzer`: Comprehensive seasonal pattern detection
- `MechanismAnalyzer`: Factor attribution and causal analysis
- `SeasonalRegressionModel`: Advanced econometric modeling

### Options Strategy Framework
- `OptionsCalculator`: Multi-method pricing (Black-Scholes, binomial, Monte Carlo)
- `GreeksCalculator`: Numerical Greeks with high precision
- `SeasonalOptionsStrategy`: Strategy optimization for seasonal patterns

### Risk Management System
- `MonteCarloEngine`: Advanced stochastic simulation
- `VaRCalculator`: Multiple VaR methodologies
- `StressTestEngine`: Scenario-based risk assessment

### Data Management
- `DataIngestionPipeline`: Multi-source data collection with failover
- `DataValidator`: Comprehensive quality control (15+ validation rules)
- `MarketDataRepository`: Optimized storage with lineage tracking

## Architecture

```
├── src/
│   ├── analysis/           # Statistical analysis modules
│   ├── data/              # Data ingestion and validation
│   ├── options/           # Options pricing and strategies
│   ├── risk/              # Risk management and Monte Carlo
│   ├── visualization/     # Charts and reporting
│   └── config.py          # Configuration management
├── specs/                 # Requirements and design docs
├── data/                  # Data storage
├── outputs/               # Analysis results
└── notebooks/             # Jupyter analysis notebooks
```

## Japanese Market Specialization

### Institutional Factors
- **Fiscal Year Effects**: March year-end rebalancing and window dressing
- **Holiday Patterns**: Golden Week, Obon, year-end trading impacts
- **Policy Cycles**: Bank of Japan meeting schedules and intervention patterns
- **Earnings Seasons**: Quarterly reporting concentration effects

### Market Microstructure
- **Trading Hours**: 9:00-11:30, 12:30-15:00 JST with break considerations
- **Settlement**: T+2 settlement with monthly SQ (Special Quotation) effects
- **Volatility Patterns**: Morning auction volatility and lunch break gaps
- **Currency Impact**: USD/JPY correlation analysis for international flows

## Output Examples

### Seasonality Analysis
```
==================================================
NIKKEI 225 SEASONALITY ANALYSIS RESULTS
==================================================

Monthly Pattern Analysis (2015-2023):
  January:   +0.8% (p=0.045) *significant*
  February:  +0.3% (p=0.234)
  March:     -1.2% (p=0.003) **highly significant**
  ...
  December:  +1.4% (p=0.012) *significant*

Strongest Seasonal Effects:
  1. March Decline: -1.2% (fiscal year-end effect)
  2. December Rally: +1.4% (year-end positioning)
  3. May Weakness: -0.9% (Golden Week disruption)
```

### Strategy Recommendations
```
==================================================
RECOMMENDED SEASONAL STRATEGIES
==================================================

Strategy 1: March Put Spread
  Entry: February 20-28
  Structure: Long 24000 Put, Short 22000 Put
  Expected P&L: +180% (based on historical patterns)
  Max Risk: ¥50,000 per contract
  Success Rate: 73% (8 of 11 years profitable)

Strategy 2: December Call Spread
  Entry: November 15-30
  Structure: Long 26000 Call, Short 28000 Call
  Expected P&L: +95% (year-end rally capture)
  Max Risk: ¥75,000 per contract
  Success Rate: 68% (7 of 10 years profitable)
```

## Data Requirements

### Primary Sources
- **Nikkei Official**: Real-time index values and constituent data
- **Japan Exchange Group (JPX)**: Official trading data and corporate actions
- **Investing.com**: Historical OHLCV data for backtesting

### Data Quality Standards
- **Accuracy Threshold**: 99.9% validation score required
- **Coverage**: Minimum 20 years historical data (≥5,000 observations)
- **Validation**: Cross-source reconciliation with automated anomaly detection
- **Updates**: Daily data refresh with intraday validation

## Performance Specifications

### Processing Capabilities
- **Historical Analysis**: 20+ years of daily data in <2 minutes
- **Monte Carlo Simulation**: 10,000 paths in <30 seconds
- **Real-time Updates**: Sub-second options pricing refresh
- **Concurrent Analysis**: Multi-strategy backtesting with parallel processing

### Accuracy Requirements
- **Options Pricing**: <0.1% deviation from market prices
- **Statistical Tests**: Proper p-value adjustments for multiple comparisons
- **Risk Calculations**: 99.5%+ accuracy for VaR backtesting
- **Seasonality Detection**: 95%+ sensitivity for true seasonal patterns

## Development

### Testing
```bash
# Run comprehensive test suite
pytest tests/ -v --cov=src

# Test specific modules
pytest tests/test_seasonality.py
pytest tests/test_options.py
pytest tests/test_risk.py
```

### Code Quality
```bash
# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

## Configuration

### Environment Variables
```bash
# Data source API keys
NIKKEI_API_KEY=your_nikkei_api_key
JPX_API_KEY=your_jpx_api_key
INVESTING_COM_API_KEY=your_investing_api_key

# Database configuration
DATABASE_URL=sqlite:///data/nk225_data.db

# Risk management
MAX_PORTFOLIO_RISK=0.02  # 2% max portfolio risk
VAR_CONFIDENCE_LEVEL=0.95
```

### Configuration File
See `src/config.py` for detailed configuration options including:
- Data source priorities and rate limits
- Statistical significance thresholds
- Options pricing parameters
- Risk management bounds
- Japanese market constants

## Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow Japanese market trading conventions
- Include comprehensive unit tests for financial calculations
- Document all statistical methodologies with academic references
- Validate against known market anomalies and historical events

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. The developers are not responsible for any financial losses incurred through use of this software.

## Acknowledgments

- **Nikkei Inc.** for index methodology and historical data access
- **Japan Exchange Group** for market structure insights
- **Quantitative Finance Community** for statistical methodology validation
- **Academic Contributors** for seasonal pattern research foundations

---

**Built with ❤️ for quantitative finance research and Japanese market analysis**
