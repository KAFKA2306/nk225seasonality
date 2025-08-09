# Coding Standards and Best Practices

## Code Style
- **Black** formatting with default settings (88 character line length)
- **flake8** linting for code quality
- **mypy** type checking for type safety
- PEP 8 compliance required

## Python Conventions
- Use type hints for all function parameters and return types
- Docstrings required for all public functions/classes (Google style preferred)
- Use descriptive variable and function names
- Prefer composition over inheritance
- Follow SOLID principles

## Financial Code Specific Standards

### Data Handling
- Always validate financial data inputs
- Handle missing data explicitly (NaN, holidays, weekends)
- Use timezone-aware datetime objects for Japanese market data (JST/Asia/Tokyo)
- Document data sources and assumptions clearly

### Statistical Analysis
- Include confidence intervals and significance tests
- Document statistical assumptions and limitations  
- Use proper sampling techniques for backtesting
- Avoid look-ahead bias in historical analysis

### Options Pricing
- Validate inputs (strikes, expiries, volatilities)
- Handle edge cases (zero time to expiry, extreme volatilities)
- Document pricing model assumptions
- Include risk metrics (Greeks) alongside prices

### Risk Management
- Set explicit risk limits and validation
- Use Monte Carlo methods with proper random seed handling
- Document risk model assumptions and limitations
- Include stress testing scenarios

## Project Structure Standards
- Modular design with clear separation of concerns
- Each module should have a single responsibility
- Use dependency injection for external services
- Comprehensive error handling and logging

## Testing Standards
- Unit tests for all business logic
- Integration tests for data pipelines
- Property-based testing for mathematical functions
- Mock external data sources in tests
- Aim for >90% test coverage

## Documentation Standards
- README files for each major component
- Inline comments for complex financial logic
- Jupyter notebooks for analysis demonstrations
- API documentation for public interfaces

## Japanese Market Considerations
- Account for Japanese fiscal year (April 1 - March 31)
- Handle Japanese market holidays correctly
- Consider time zone differences in data processing
- Document market-specific institutional factors