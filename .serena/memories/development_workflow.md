# Development Workflow and Commands

## Project Status
- **Current State**: Early stage - no source code yet, only specifications
- **Package Management**: Not yet established (no requirements.txt, pyproject.toml, or setup.py)
- **Git Repository**: Not yet initialized
- **Testing Framework**: Not yet established

## Recommended Development Setup Commands

### Initial Project Setup (when implementing)
```bash
# Initialize git repository
git init
git add .
git commit -m "Initial project structure"

# Create Python package management
# Option 1: requirements.txt approach
touch requirements.txt

# Option 2: Modern pyproject.toml approach
touch pyproject.toml
```

### Package Management (to be established)
```bash
# Install dependencies (when requirements.txt exists)
pip install -r requirements.txt

# Or with pyproject.toml
pip install -e .
```

### Testing Commands (to be established)
```bash
# Run tests (framework TBD - pytest recommended)
pytest

# With coverage
pytest --cov=src/

# Specific test modules
pytest tests/test_seasonality.py
```

### Code Quality Commands (to be established)
```bash
# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/
# Or
pylint src/ tests/

# Type checking
mypy src/

# Import sorting
isort src/ tests/
```

### Data Pipeline Commands (to be implemented)
```bash
# Data collection
python -m src.data.collect_nikkei_data

# Statistical analysis
python -m src.analysis.seasonality_analyzer

# Strategy backtesting  
python -m src.strategies.backtest_options
```

## Specialized Agent Usage
- Use `nikkei-seasonality-analyst` agent for statistical analysis and options strategy development
- This agent has specialized knowledge of Japanese market patterns and quantitative finance