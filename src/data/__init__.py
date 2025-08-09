"""
Data management package for Nikkei 225 seasonality analysis.

This package provides comprehensive data collection, validation, and storage
capabilities for multiple financial data sources with robust error handling
and data quality assurance.
"""

from .ingestion import DataIngestionPipeline, DataSource
from .validation import DataValidator, ValidationResult
from .repository import MarketDataRepository

__all__ = [
    'DataIngestionPipeline',
    'DataSource', 
    'DataValidator',
    'ValidationResult',
    'MarketDataRepository'
]