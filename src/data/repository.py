"""
Data repository module for market data storage and retrieval.

This module provides a repository pattern implementation for accessing
market data with caching, query optimization, and data lineage tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import logging
from contextlib import contextmanager
import json

from ..config import SystemConfig, get_logger, JapaneseMarketConstants


class MarketDataRepository:
    """Repository for market data storage and retrieval."""
    
    def __init__(self, config: SystemConfig):
        """Initialize repository with database connection."""
        self.config = config
        self.logger = get_logger("data.repository")
        
        # Use SQLite for local development, can be extended to PostgreSQL
        self.db_path = config.data_dir / "nikkei_data.db"
        self._ensure_database_schema()
        
        # In-memory cache for frequently accessed data
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamps = {}
    
    def _ensure_database_schema(self):
        """Ensure database schema exists."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS nikkei_daily_data (
            trade_date TEXT PRIMARY KEY,
            open_price REAL NOT NULL,
            high_price REAL NOT NULL,
            low_price REAL NOT NULL,
            close_price REAL NOT NULL,
            volume INTEGER,
            adjusted_close REAL NOT NULL,
            data_source TEXT NOT NULL,
            data_quality_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS data_validation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            source_name TEXT NOT NULL,
            validation_rule TEXT NOT NULL,
            status TEXT NOT NULL,
            details TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS data_lineage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            operation TEXT NOT NULL,
            source_data TEXT,
            result_data TEXT,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_trade_date ON nikkei_daily_data(trade_date);
        CREATE INDEX IF NOT EXISTS idx_data_source ON nikkei_daily_data(data_source);
        CREATE INDEX IF NOT EXISTS idx_validation_date ON data_validation_log(trade_date);
        """
        
        with self._get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context management."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def store_data(self, data: pd.DataFrame, source: str = "unknown") -> bool:
        """Store market data in the database."""
        try:
            with self._get_connection() as conn:
                for idx, row in data.iterrows():
                    # Convert timestamp to string
                    trade_date = idx.strftime('%Y-%m-%d') if isinstance(idx, pd.Timestamp) else str(idx)
                    
                    # Prepare data for insertion
                    record = {
                        'trade_date': trade_date,
                        'open_price': float(row.get('open_price', 0)),
                        'high_price': float(row.get('high_price', 0)),
                        'low_price': float(row.get('low_price', 0)),
                        'close_price': float(row.get('close_price', 0)),
                        'volume': int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
                        'adjusted_close': float(row.get('adjusted_close', row.get('close_price', 0))),
                        'data_source': source,
                        'data_quality_score': float(row.get('data_quality_score', 1.0)),
                        'updated_at': datetime.now().isoformat()
                    }
                    
                    # Use INSERT OR REPLACE to handle duplicates
                    conn.execute("""
                        INSERT OR REPLACE INTO nikkei_daily_data 
                        (trade_date, open_price, high_price, low_price, close_price, 
                         volume, adjusted_close, data_source, data_quality_score, updated_at)
                        VALUES (:trade_date, :open_price, :high_price, :low_price, :close_price,
                                :volume, :adjusted_close, :data_source, :data_quality_score, :updated_at)
                    """, record)
                
                conn.commit()
                
            self.logger.info(f"Stored {len(data)} records from source: {source}")
            self._invalidate_cache()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store data: {e}")
            return False
    
    def get_historical_data(self, 
                           start_date: datetime, 
                           end_date: datetime,
                           quality_threshold: float = 0.8) -> pd.DataFrame:
        """Retrieve validated historical data."""
        cache_key = f"historical_{start_date.date()}_{end_date.date()}_{quality_threshold}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        query = """
            SELECT trade_date, open_price, high_price, low_price, close_price, 
                   volume, adjusted_close, data_source, data_quality_score
            FROM nikkei_daily_data 
            WHERE trade_date BETWEEN ? AND ? 
            AND data_quality_score >= ?
            ORDER BY trade_date
        """
        
        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(
                    query, 
                    conn, 
                    params=[start_date.strftime('%Y-%m-%d'), 
                           end_date.strftime('%Y-%m-%d'), 
                           quality_threshold]
                )
            
            if not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.set_index('trade_date')
                
                # Calculate returns
                df['returns'] = df['adjusted_close'].pct_change()
                
                # Cache the result
                self._cache[cache_key] = df
                self._cache_timestamps[cache_key] = datetime.now()
            
            self.logger.info(f"Retrieved {len(df)} records from {start_date.date()} to {end_date.date()}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve historical data: {e}")
            return pd.DataFrame()
    
    def get_seasonal_subset(self, 
                           months: List[int], 
                           years: int = 20,
                           quality_threshold: float = 0.8) -> pd.DataFrame:
        """Extract seasonal data for analysis."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Get all data for the period
        data = self.get_historical_data(start_date, end_date, quality_threshold)
        
        if data.empty:
            return data
        
        # Filter by months
        seasonal_data = data[data.index.month.isin(months)]
        
        self.logger.info(f"Extracted {len(seasonal_data)} seasonal records for months {months}")
        return seasonal_data
    
    def get_latest_data(self, days: int = 30) -> pd.DataFrame:
        """Get the most recent data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.get_historical_data(start_date, end_date)
    
    def get_data_coverage(self) -> Dict[str, Any]:
        """Get information about data coverage and quality."""
        query = """
            SELECT 
                MIN(trade_date) as earliest_date,
                MAX(trade_date) as latest_date,
                COUNT(*) as total_records,
                AVG(data_quality_score) as avg_quality,
                COUNT(DISTINCT data_source) as source_count
            FROM nikkei_daily_data
        """
        
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(query)
                result = cursor.fetchone()
                
                if result:
                    coverage = {
                        'earliest_date': result['earliest_date'],
                        'latest_date': result['latest_date'],
                        'total_records': result['total_records'],
                        'avg_quality_score': result['avg_quality'],
                        'source_count': result['source_count']
                    }
                    
                    # Get source breakdown
                    source_query = """
                        SELECT data_source, COUNT(*) as count, AVG(data_quality_score) as avg_quality
                        FROM nikkei_daily_data 
                        GROUP BY data_source
                    """
                    source_df = pd.read_sql_query(source_query, conn)
                    coverage['sources'] = source_df.to_dict('records')
                    
                    return coverage
                
        except Exception as e:
            self.logger.error(f"Failed to get data coverage: {e}")
        
        return {}
    
    def log_validation_result(self, 
                             trade_date: datetime,
                             source_name: str,
                             validation_rule: str,
                             status: str,
                             details: Dict[str, Any]):
        """Log validation results."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO data_validation_log 
                    (trade_date, source_name, validation_rule, status, details)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    trade_date.strftime('%Y-%m-%d'),
                    source_name,
                    validation_rule,
                    status,
                    json.dumps(details)
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to log validation result: {e}")
    
    def get_validation_history(self, 
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get validation history."""
        query = """
            SELECT trade_date, source_name, validation_rule, status, details, created_at
            FROM data_validation_log
        """
        params = []
        
        if start_date or end_date:
            query += " WHERE "
            conditions = []
            
            if start_date:
                conditions.append("trade_date >= ?")
                params.append(start_date.strftime('%Y-%m-%d'))
            
            if end_date:
                conditions.append("trade_date <= ?")
                params.append(end_date.strftime('%Y-%m-%d'))
            
            query += " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC"
        
        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df['created_at'] = pd.to_datetime(df['created_at'])
                return df
                
        except Exception as e:
            self.logger.error(f"Failed to get validation history: {e}")
            return pd.DataFrame()
    
    def log_data_lineage(self,
                        trade_date: datetime,
                        operation: str,
                        source_data: Dict[str, Any],
                        result_data: Dict[str, Any],
                        metadata: Dict[str, Any]):
        """Log data transformation lineage."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO data_lineage 
                    (trade_date, operation, source_data, result_data, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    trade_date.strftime('%Y-%m-%d'),
                    operation,
                    json.dumps(source_data),
                    json.dumps(result_data),
                    json.dumps(metadata)
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to log data lineage: {e}")
    
    def export_data(self, 
                   start_date: datetime,
                   end_date: datetime,
                   format: str = 'csv') -> Optional[str]:
        """Export data to file."""
        data = self.get_historical_data(start_date, end_date)
        
        if data.empty:
            self.logger.warning("No data to export")
            return None
        
        filename = f"nikkei_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        if format.lower() == 'csv':
            filepath = self.config.output_dir / f"{filename}.csv"
            data.to_csv(filepath)
        elif format.lower() == 'json':
            filepath = self.config.output_dir / f"{filename}.json"
            data.to_json(filepath, date_format='iso', indent=2)
        elif format.lower() == 'parquet':
            filepath = self.config.output_dir / f"{filename}.parquet"
            data.to_parquet(filepath)
        else:
            self.logger.error(f"Unsupported export format: {format}")
            return None
        
        self.logger.info(f"Exported data to {filepath}")
        return str(filepath)
    
    def get_missing_dates(self, 
                         start_date: datetime, 
                         end_date: datetime) -> List[datetime]:
        """Get list of missing trading dates in the range."""
        # Get existing dates
        existing_data = self.get_historical_data(start_date, end_date)
        existing_dates = set(existing_data.index.date)
        
        # Generate expected trading dates (rough approximation)
        expected_dates = []
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends (rough approximation)
            if current_date.weekday() < 5:
                expected_dates.append(current_date.date())
            current_date += timedelta(days=1)
        
        # Find missing dates
        missing_dates = [
            datetime.combine(date, datetime.min.time())
            for date in set(expected_dates) - existing_dates
        ]
        
        return sorted(missing_dates)
    
    def cleanup_old_data(self, days_to_keep: int = 365 * 5):
        """Clean up old data beyond retention period."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        try:
            with self._get_connection() as conn:
                # Delete old data
                cursor = conn.execute(
                    "DELETE FROM nikkei_daily_data WHERE trade_date < ?",
                    (cutoff_date.strftime('%Y-%m-%d'),)
                )
                deleted_count = cursor.rowcount
                
                # Delete old validation logs
                cursor = conn.execute(
                    "DELETE FROM data_validation_log WHERE trade_date < ?",
                    (cutoff_date.strftime('%Y-%m-%d'),)
                )
                
                # Delete old lineage logs
                cursor = conn.execute(
                    "DELETE FROM data_lineage WHERE trade_date < ?",
                    (cutoff_date.strftime('%Y-%m-%d'),)
                )
                
                conn.commit()
                self.logger.info(f"Cleaned up {deleted_count} old records")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        
        cache_time = self._cache_timestamps.get(cache_key)
        if not cache_time:
            return False
        
        return (datetime.now() - cache_time).seconds < self._cache_ttl
    
    def _invalidate_cache(self):
        """Invalidate all cached data."""
        self._cache.clear()
        self._cache_timestamps.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with self._get_connection() as conn:
                stats = {}
                
                # Record counts
                cursor = conn.execute("SELECT COUNT(*) FROM nikkei_daily_data")
                stats['total_records'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM data_validation_log")
                stats['validation_logs'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM data_lineage")
                stats['lineage_logs'] = cursor.fetchone()[0]
                
                # Database size
                stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
                
                # Quality distribution
                cursor = conn.execute("""
                    SELECT 
                        AVG(data_quality_score) as avg_quality,
                        MIN(data_quality_score) as min_quality,
                        MAX(data_quality_score) as max_quality
                    FROM nikkei_daily_data
                """)
                quality_stats = cursor.fetchone()
                if quality_stats:
                    stats['quality'] = {
                        'average': quality_stats[0],
                        'minimum': quality_stats[1],
                        'maximum': quality_stats[2]
                    }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}