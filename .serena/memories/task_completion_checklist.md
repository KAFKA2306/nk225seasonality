# Task Completion Checklist

## Pre-Implementation Checklist
- [ ] Understand the specific requirements clearly
- [ ] Review relevant project memories and documentation
- [ ] Check if nikkei-seasonality-analyst agent should be used
- [ ] Identify affected modules and dependencies
- [ ] Plan the implementation approach

## During Implementation Checklist

### Code Quality
- [ ] Follow established coding standards
- [ ] Add appropriate type hints
- [ ] Write descriptive docstrings
- [ ] Include error handling
- [ ] Add logging statements where appropriate
- [ ] Handle Japanese market specifics correctly

### Financial Code Specific
- [ ] Validate all financial data inputs
- [ ] Handle missing data and edge cases
- [ ] Use timezone-aware datetime objects (JST)
- [ ] Document assumptions and limitations
- [ ] Include confidence intervals for statistics
- [ ] Avoid look-ahead bias in analysis

### Testing
- [ ] Write unit tests for new functionality
- [ ] Test edge cases and error conditions
- [ ] Mock external dependencies
- [ ] Run existing tests to ensure no regressions
- [ ] Achieve adequate test coverage

## Post-Implementation Checklist

### Code Quality Validation
- [ ] Run `black src/ tests/` for formatting
- [ ] Run `flake8 src/ tests/` for linting
- [ ] Run `mypy src/` for type checking
- [ ] All code quality checks pass

### Testing Validation
- [ ] Run `pytest` - all tests pass
- [ ] Run `pytest --cov=src` - coverage adequate
- [ ] Manual testing of new functionality
- [ ] Integration testing if applicable

### Documentation
- [ ] Update relevant docstrings
- [ ] Update README if public API changed
- [ ] Update specs/ if requirements changed
- [ ] Add usage examples if needed

### Final Verification
- [ ] Code follows project architecture patterns
- [ ] No sensitive data or credentials in code
- [ ] Logging is appropriate and structured
- [ ] Performance considerations addressed
- [ ] Japanese market specifics handled correctly

## Specialized Tasks Checklist

### Statistical Analysis Tasks
- [ ] Use nikkei-seasonality-analyst agent
- [ ] Include significance testing
- [ ] Document statistical assumptions
- [ ] Provide confidence intervals
- [ ] Validate sample sizes
- [ ] Check for multiple comparisons issues

### Options Strategy Tasks  
- [ ] Use nikkei-seasonality-analyst agent
- [ ] Validate pricing inputs
- [ ] Calculate Greeks
- [ ] Include risk metrics
- [ ] Document strategy assumptions
- [ ] Provide backtesting results

### Data Processing Tasks
- [ ] Handle Japanese market holidays
- [ ] Validate data quality
- [ ] Handle missing data appropriately
- [ ] Use proper timezone handling
- [ ] Document data sources
- [ ] Include data lineage information

## Deployment Checklist
- [ ] All tests passing in clean environment
- [ ] Dependencies properly specified
- [ ] Configuration properly externalized
- [ ] Logging configured correctly
- [ ] Error monitoring in place
- [ ] Performance benchmarks established