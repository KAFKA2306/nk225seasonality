# Suggested Commands and Development Workflow

## Testing Commands
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test files
pytest tests/test_seasonality.py
pytest tests/test_options.py
pytest tests/test_data_ingestion.py
```

## Code Quality Commands
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Development Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run main analysis pipeline
python main.py

# Start Jupyter notebook for analysis
jupyter notebook

# Run specific analysis modules
python -m src.analysis.seasonality
python -m src.options.strategies
python -m src.risk.monte_carlo
```

## Data Management Commands
```bash
# Run data ingestion
python -m src.data.ingestion

# Validate data quality
python -m src.data.validation

# Check database schema
python -c "from src.data.repository import get_engine; print(get_engine().table_names())"
```

## Logging and Monitoring
```bash
# View recent logs
tail -f logs/nikkei_analysis.log
tail -f logs/errors.log

# Clear logs
rm logs/*.log
```

## Analysis Commands
```bash
# Run seasonality analysis
python -m src.analysis.seasonality

# Generate visualizations
python -m src.visualization.seasonality_viz

# Calculate option strategies
python -m src.options.strategies

# Run risk analysis
python -m src.risk.monte_carlo
```

## Database Operations
- Database file: `data/nikkei_data.db`
- Schema managed via SQLAlchemy
- Backup: `cp data/nikkei_data.db data/nikkei_data_backup.db`