# Coding Standards and Conventions

## Code Style Guidelines

### Python Style Standards
- **PEP 8** compliance for all Python code
- **Black** for automated code formatting
- **Maximum line length**: 88 characters (Black default)
- **Import organization**: Use isort for consistent import ordering

### Naming Conventions
- **Classes**: PascalCase (e.g., `SeasonalityAnalyzer`, `OptionsCalculator`)
- **Functions/Methods**: snake_case (e.g., `test_monthly_patterns`, `calculate_greeks`)
- **Variables**: snake_case (e.g., `monthly_returns`, `significance_level`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `SIGNIFICANCE_LEVEL`, `DATA_SOURCES`)
- **Private methods**: Leading underscore (e.g., `_calculate_payoff`)

### Type Hints
- **Mandatory** for all function signatures
- Use `from typing import Dict, List, Any, Optional` as needed
- Example:
```python
def analyze_seasonality(data: pd.DataFrame, 
                       significance: float = 0.05) -> Dict[str, Any]:
```

### Docstrings
- **Google style** docstrings for all classes and functions
- Include parameter types, return types, and examples for complex functions
- Example:
```python
def calculate_black_scholes(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes option price.
    
    Args:
        S: Current stock price
        K: Strike price  
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        
    Returns:
        Option price as float
        
    Example:
        >>> calculate_black_scholes(100, 105, 0.25, 0.05, 0.2)
        2.13
    """
```

### Error Handling
- Use specific exception types rather than generic `Exception`
- Implement custom exceptions for domain-specific errors:
```python
class DataValidationError(Exception):
    """Raised when data quality checks fail"""
    
class InsufficientDataError(Exception):
    """Raised when insufficient data for statistical analysis"""
```

### Logging Standards
- Use Python's `logging` module, not print statements
- Configure appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Include contextual information in log messages
- Example:
```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"Analyzing seasonality for {len(data)} observations")
```

## Financial Analysis Standards

### Statistical Rigor
- Always include confidence intervals and p-values
- Use proper multiple comparison corrections (Bonferroni, FDR)
- Document assumptions and limitations of statistical tests
- Validate results with out-of-sample testing

### Data Validation
- Implement comprehensive data quality checks
- Handle missing values explicitly (never ignore silently)
- Cross-validate data from multiple sources
- Maintain audit trails for all data transformations

### Performance Requirements
- Optimize for large datasets (20+ years of daily data)
- Use vectorized operations (pandas/numpy) over loops
- Implement chunked processing for memory management
- Cache expensive calculations appropriately

## Project Structure
```
src/
├── data/           # Data collection and processing
├── analysis/       # Statistical analysis modules  
├── strategies/     # Options strategy implementation
├── risk/          # Risk management and Monte Carlo
├── visualization/ # Charting and reporting
├── utils/         # Utility functions
└── config/        # Configuration management
```

## Testing Standards
- **Minimum 80% code coverage**
- Unit tests for all statistical functions
- Integration tests for data pipelines
- Mock external data sources in tests
- Property-based testing for mathematical functions