# Technology Stack and Dependencies

## Core Scientific Computing
- **numpy** (>=1.21.0) - Numerical computing foundation
- **pandas** (>=1.3.0) - Data manipulation and analysis
- **scipy** (>=1.7.0) - Scientific computing algorithms

## Statistical Analysis
- **statsmodels** (>=0.12.0) - Statistical modeling and econometrics
- **scikit-learn** (>=1.0.0) - Machine learning algorithms

## Quantitative Finance
- **quantlib** (>=1.26) - Options pricing and derivatives (optional, can be challenging to install)
- Custom implementations available as alternatives

## Data Visualization  
- **matplotlib** (>=3.4.0) - Basic plotting
- **seaborn** (>=0.11.0) - Statistical visualization
- **plotly** (>=5.0.0) - Interactive charts

## Data Sources & APIs
- **aiohttp** (>=3.8.0) - Async HTTP client
- **requests** (>=2.25.0) - HTTP library
- **yfinance** (optional) - Yahoo Finance API
- **alpha-vantage** (optional) - Financial data API
- **quandl** (optional) - Economic data API

## Database & Storage
- **sqlalchemy** (>=1.4.0) - Database ORM
- **openpyxl** (>=3.0.0) - Excel file support
- **xlsxwriter** (>=3.0.0) - Excel writing
- Current: SQLite database (nikkei_data.db)

## Development & Testing
- **pytest** (>=6.0.0) - Testing framework
- **pytest-asyncio** (>=0.15.0) - Async testing
- **pytest-cov** (>=2.12.0) - Coverage reporting
- **black** (>=21.0.0) - Code formatting
- **flake8** (>=3.9.0) - Linting
- **mypy** (>=0.910) - Type checking

## Utilities
- **python-dotenv** (>=0.19.0) - Environment management
- **structlog** (>=21.1.0) - Structured logging
- **tqdm** (>=4.62.0) - Progress bars
- **pytz** (>=2021.1) - Timezone handling
- **numba** (>=0.54.0) - JIT compilation (optional)
- **cython** (>=0.29.0) - C extensions (optional)

## Jupyter Support
- **jupyter** (>=1.0.0) - Notebook environment
- **ipykernel** (>=6.0.0) - Python kernel