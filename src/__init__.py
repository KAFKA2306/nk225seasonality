"""
Nikkei 225 Seasonality Analysis Package

A comprehensive quantitative finance research platform for analyzing seasonal patterns
in the Nikkei 225 index and developing options strategies based on seasonal effects.

This package provides:
- Multi-source data collection and validation
- Statistical analysis of seasonal patterns
- Options pricing and strategy development  
- Monte Carlo risk assessment
- Comprehensive visualization capabilities

Example usage:
    from src.pipeline import AnalysisPipeline
    from src.config import SystemConfig
    from datetime import datetime
    
    # Initialize pipeline with default configuration
    config = SystemConfig()
    pipeline = AnalysisPipeline(config)
    
    # Run full analysis
    results = await pipeline.run_full_analysis(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31)
    )
"""

from .config import SystemConfig, setup_logging, get_logger
from .pipeline import AnalysisPipeline

# Core analysis components
from .analysis import SeasonalityAnalyzer, SeasonalRegressionModel, MechanismAnalyzer
from .options import OptionsCalculator, SeasonalOptionsStrategy, StrategyBacktester
from .risk import MonteCarloEngine, VaRCalculator
from .data import DataIngestionPipeline, DataValidator, MarketDataRepository
from .visualization import SeasonalityVisualizer, OptionsVisualizer, RiskVisualizer

__version__ = "1.0.0"
__author__ = "Nikkei 225 Seasonality Analysis Team"
__email__ = "contact@nikkei-analysis.com"

__all__ = [
    # Main components
    'SystemConfig',
    'AnalysisPipeline',
    'setup_logging',
    'get_logger',
    
    # Analysis modules
    'SeasonalityAnalyzer',
    'SeasonalRegressionModel', 
    'MechanismAnalyzer',
    
    # Options modules
    'OptionsCalculator',
    'SeasonalOptionsStrategy',
    'StrategyBacktester',
    
    # Risk modules
    'MonteCarloEngine',
    'VaRCalculator',
    
    # Data modules
    'DataIngestionPipeline',
    'DataValidator',
    'MarketDataRepository',
    
    # Visualization modules
    'SeasonalityVisualizer',
    'OptionsVisualizer',
    'RiskVisualizer'
]