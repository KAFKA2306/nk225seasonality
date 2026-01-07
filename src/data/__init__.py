"""
Data management package for Nikkei 225 seasonality analysis.

This package provides comprehensive data collection, validation, and storage
capabilities for multiple financial data sources with robust error handling
and data quality assurance.
"""

from .ingestion import DataIngestionPipeline, DataSource
from .repository import MarketDataRepository
from .validation import DataValidator, ValidationResult

__all__ = [
    "DataIngestionPipeline",
    "DataSource",
    "DataValidator",
    "ValidationResult",
    "MarketDataRepository",
]
