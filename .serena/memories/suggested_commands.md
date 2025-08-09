# Suggested Commands for Development

## Current Project State
⚠️ **Note**: This project is in early stage with no source code yet. These commands will be relevant once implementation begins.

## Essential Development Commands

### Project Initialization
```bash
git init                          # Initialize git repository
python -m venv venv              # Create virtual environment  
source venv/bin/activate         # Activate virtual environment (Linux)
```

### Package Management
```bash
pip install -r requirements.txt  # Install dependencies (once created)
pip freeze > requirements.txt    # Save current dependencies
pip install -e .                # Install project in development mode
```

### Code Quality (once established)
```bash
black .                         # Code formatting
flake8 .                       # Linting  
mypy src/                      # Type checking
pytest                         # Run tests
pytest --cov=src/             # Run tests with coverage
```

### Data Operations (to be implemented)
```bash
python -m src.data.collect     # Collect historical data
python -m src.analysis.main    # Run seasonality analysis
python -m src.strategies.backtest  # Backtest strategies
```

### System Utilities (Linux)
```bash
ls -la                         # List files with details
find . -name "*.py"           # Find Python files
grep -r "pattern" src/        # Search in source code
du -sh data/                  # Check data directory size
```

## Analysis Pipeline Commands (future)
```bash
# Statistical analysis workflow
python analyze_seasonality.py --period 20y --significance 0.05
python generate_strategies.py --month 3 --strategy put_spread  
python backtest_portfolio.py --start 2000 --end 2023
```

## Docker Commands (if containerized)
```bash
docker build -t nikkei-analysis .
docker run -v $(pwd)/data:/app/data nikkei-analysis
```

## Notes
- Use the `nikkei-seasonality-analyst` agent for complex statistical analysis tasks
- All financial analysis should maintain academic rigor with proper statistical testing
- Data validation is critical given the requirements for 99.9% accuracy