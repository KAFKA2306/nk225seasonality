# Development Workflow and Best Practices

## Git Workflow
- **Main branch**: `main` (current branch)
- Feature branches for new functionality
- Commit messages should be descriptive and reference issues
- Use conventional commits format when possible

## Development Process

### 1. Before Starting Work
- Update from main branch
- Create feature branch if needed
- Review relevant specs in `specs/` directory
- Check current project memories and documentation

### 2. Development Cycle
- Write failing tests first (TDD approach)
- Implement functionality
- Run code quality checks (`black`, `flake8`, `mypy`)
- Run tests (`pytest`)
- Update documentation if needed

### 3. Specialized Agent Usage
**Always use the nikkei-seasonality-analyst agent for:**
- Statistical analysis tasks
- Market data analysis
- Options strategy development
- Risk management implementations
- Japanese market specific considerations

### 4. Code Review Checklist
- [ ] All tests passing
- [ ] Code formatted with black
- [ ] No linting errors
- [ ] Type hints added
- [ ] Docstrings updated
- [ ] Log statements appropriate
- [ ] Error handling implemented
- [ ] Japanese market specifics considered

## Project Architecture

### Core Modules
- **data/**: Data ingestion, validation, repository pattern
- **analysis/**: Seasonality analysis, market mechanisms
- **options/**: Pricing calculations, strategy implementations
- **risk/**: VaR calculations, Monte Carlo simulations
- **visualization/**: Charts and analysis outputs
- **utils/**: Common utilities and helpers

### Key Design Patterns
- Repository pattern for data access
- Strategy pattern for options strategies
- Pipeline pattern for analysis workflows
- Observer pattern for logging and monitoring

## Data Flow
1. **Ingestion**: Raw data → Database (SQLite)
2. **Validation**: Quality checks and cleaning
3. **Analysis**: Statistical analysis and pattern detection
4. **Strategy**: Options strategy development
5. **Risk**: Risk assessment and backtesting
6. **Visualization**: Results presentation

## Environment Management
- Use `.env` files for configuration (not committed)
- Database: `data/nikkei_data.db`
- Logs: `logs/` directory
- Outputs: `outputs/` directory
- Notebooks: `notebooks/` directory for exploration

## Academic Rigor Requirements
- Proper statistical significance testing
- Validation of assumptions
- Out-of-sample testing
- Documentation of methodology
- Reproducible analysis pipelines
- Citation of relevant literature

## Japanese Market Considerations
- Trading hours: 9:00-11:30, 12:30-15:00 JST
- Fiscal year-end effects (March 31)
- Golden Week and other holidays
- Institutional investment patterns
- Currency hedging considerations