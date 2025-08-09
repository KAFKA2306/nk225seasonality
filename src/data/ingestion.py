"""
Data ingestion pipeline for collecting Nikkei 225 data from multiple sources.

This module implements a robust data collection framework that can handle
multiple data sources with proper error handling, rate limiting, and
data reconciliation capabilities.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import time
import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import SystemConfig, DataSourceConfig, get_logger
from .validation import DataValidator


@dataclass
class DataFetchResult:
    """Result of a data fetch operation."""
    
    source_name: str
    success: bool
    data: Optional[pd.DataFrame]
    error: Optional[str]
    timestamp: datetime
    latency_ms: float


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, name: str, config: DataSourceConfig):
        self.name = name
        self.config = config
        self.logger = get_logger(f"data.source.{name}")
        self._rate_limiter = RateLimiter(config.rate_limit)
    
    @abstractmethod
    async def fetch_data(self, 
                        start_date: datetime, 
                        end_date: datetime) -> pd.DataFrame:
        """Fetch data for the specified date range."""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate that the data source is accessible."""
        pass
    
    async def _make_request(self, 
                           session: aiohttp.ClientSession,
                           url: str,
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request with rate limiting and retry logic."""
        await self._rate_limiter.acquire()
        
        for attempt in range(self.config.retry_attempts):
            try:
                async with session.get(
                    url, 
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limited
                        wait_time = 60 * (attempt + 1)
                        self.logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        response.raise_for_status()
                        
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt == self.config.retry_attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
                
            except aiohttp.ClientError as e:
                self.logger.warning(f"Request failed on attempt {attempt + 1}: {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        
        raise Exception(f"Max retries ({self.config.retry_attempts}) exceeded")


class NikkeiOfficialSource(DataSource):
    """Data source for official Nikkei data."""
    
    async def fetch_data(self, 
                        start_date: datetime, 
                        end_date: datetime) -> pd.DataFrame:
        """Fetch data from Nikkei official API."""
        if not self.config.api_key:
            raise ValueError("API key required for Nikkei official source")
        
        url = f"{self.config.base_url}historical/daily"
        params = {
            'symbol': 'NK225',
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'api_key': self.config.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            data = await self._make_request(session, url, params)
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            df['trade_date'] = pd.to_datetime(df['date'])
            df['source'] = self.name
            
            # Standardize column names
            df = df.rename(columns={
                'open': 'open_price',
                'high': 'high_price', 
                'low': 'low_price',
                'close': 'close_price',
                'adjusted_close': 'adjusted_close'
            })
            
            return df[['trade_date', 'open_price', 'high_price', 
                      'low_price', 'close_price', 'volume', 
                      'adjusted_close', 'source']].set_index('trade_date')
    
    def validate_connection(self) -> bool:
        """Test connection to Nikkei API."""
        try:
            # Simple connection test - fetch latest quote
            import requests
            response = requests.get(
                f"{self.config.base_url}quote/NK225",
                params={'api_key': self.config.api_key},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            return False


class InvestingComSource(DataSource):
    """Data source for Investing.com data."""
    
    async def fetch_data(self, 
                        start_date: datetime, 
                        end_date: datetime) -> pd.DataFrame:
        """Fetch data from Investing.com API."""
        url = f"{self.config.base_url}historical"
        params = {
            'symbol': 'indices/nikkei-225',
            'resolution': 'daily',
            'from': int(start_date.timestamp()),
            'to': int(end_date.timestamp())
        }
        
        headers = {
            'X-API-KEY': self.config.api_key,
            'User-Agent': 'Nikkei-Analysis/1.0'
        }
        
        async with aiohttp.ClientSession(headers=headers) as session:
            data = await self._make_request(session, url, params)
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'trade_date': pd.to_datetime(data['t'], unit='s'),
                'open_price': data['o'],
                'high_price': data['h'],
                'low_price': data['l'],
                'close_price': data['c'],
                'volume': data['v'],
                'source': self.name
            })
            
            # Investing.com doesn't provide adjusted close, use close
            df['adjusted_close'] = df['close_price']
            
            return df.set_index('trade_date')
    
    def validate_connection(self) -> bool:
        """Test connection to Investing.com API."""
        try:
            import requests
            response = requests.get(
                f"{self.config.base_url}quote",
                params={'symbol': 'indices/nikkei-225'},
                headers={'X-API-KEY': self.config.api_key},
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False


class JPXOfficialSource(DataSource):
    """Data source for Japan Exchange Group official data."""
    
    async def fetch_data(self, 
                        start_date: datetime, 
                        end_date: datetime) -> pd.DataFrame:
        """Fetch data from JPX official API."""
        url = f"{self.config.base_url}market-data/indices/daily"
        params = {
            'index_code': 'NK225',
            'date_from': start_date.strftime('%Y%m%d'),
            'date_to': end_date.strftime('%Y%m%d')
        }
        
        async with aiohttp.ClientSession() as session:
            data = await self._make_request(session, url, params)
            
            # Parse JPX format
            records = []
            for item in data['indices']:
                if item['index_code'] == 'NK225':
                    records.append({
                        'trade_date': pd.to_datetime(item['date'], format='%Y%m%d'),
                        'open_price': float(item['open_value']),
                        'high_price': float(item['high_value']),
                        'low_price': float(item['low_value']),
                        'close_price': float(item['close_value']),
                        'volume': int(item.get('volume', 0)),
                        'adjusted_close': float(item['close_value']),
                        'source': self.name
                    })
            
            df = pd.DataFrame(records)
            return df.set_index('trade_date') if not df.empty else pd.DataFrame()
    
    def validate_connection(self) -> bool:
        """Test connection to JPX API."""
        try:
            import requests
            response = requests.get(
                f"{self.config.base_url}market-data/indices/current",
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False


class LocalFileSource(DataSource):
    """Data source for local CSV files."""
    
    def __init__(self, name: str, config: DataSourceConfig, file_path: str):
        super().__init__(name, config)
        self.file_path = Path(file_path)
    
    async def fetch_data(self, 
                        start_date: datetime, 
                        end_date: datetime) -> pd.DataFrame:
        """Load data from local CSV file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        df = pd.read_csv(self.file_path)
        df['trade_date'] = pd.to_datetime(df['Date'])
        df['source'] = self.name
        
        # Standardize column names
        column_mapping = {
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price', 
            'Close': 'close_price',
            'Adj Close': 'adjusted_close',
            'Volume': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        df = df.set_index('trade_date')
        
        # Filter by date range
        mask = (df.index >= start_date) & (df.index <= end_date)
        return df[mask]
    
    def validate_connection(self) -> bool:
        """Check if local file exists and is readable."""
        return self.file_path.exists() and self.file_path.is_file()


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, requests_per_hour: int):
        self.requests_per_hour = requests_per_hour
        self.requests_per_second = requests_per_hour / 3600
        self.min_interval = 1.0 / self.requests_per_second
        self.last_request_time = 0
    
    async def acquire(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)
        
        self.last_request_time = time.time()


class DataReconciler:
    """Reconciles data from multiple sources."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def reconcile_sources(self, source_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Reconcile data from multiple sources into a single dataset."""
        if not source_data:
            return pd.DataFrame()
        
        # Sort sources by priority (assume priority is embedded in source name order)
        sorted_sources = list(source_data.keys())
        
        # Start with highest priority source
        primary_source = sorted_sources[0]
        reconciled = source_data[primary_source].copy()
        
        # Fill gaps with data from other sources
        for source_name in sorted_sources[1:]:
            source_df = source_data[source_name]
            
            # Find missing dates in primary source
            missing_dates = source_df.index.difference(reconciled.index)
            
            if len(missing_dates) > 0:
                self.logger.info(f"Filling {len(missing_dates)} missing dates from {source_name}")
                reconciled = pd.concat([reconciled, source_df.loc[missing_dates]])
        
        # Add data quality score
        reconciled['data_quality_score'] = self._calculate_quality_score(reconciled)
        
        return reconciled.sort_index()
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate data quality score for each record."""
        scores = pd.Series(1.0, index=df.index)
        
        # Check for missing values
        missing_penalty = df.isnull().sum(axis=1) * 0.1
        scores -= missing_penalty
        
        # Check for price inconsistencies
        price_cols = ['open_price', 'high_price', 'low_price', 'close_price']
        if all(col in df.columns for col in price_cols):
            # High should be >= max(open, close) and low should be <= min(open, close)
            invalid_high = df['high_price'] < df[['open_price', 'close_price']].max(axis=1)
            invalid_low = df['low_price'] > df[['open_price', 'close_price']].min(axis=1)
            
            scores[invalid_high | invalid_low] -= 0.2
        
        # Check for extreme values (>5 sigma)
        for col in ['open_price', 'high_price', 'low_price', 'close_price']:
            if col in df.columns:
                returns = df[col].pct_change()
                z_scores = np.abs((returns - returns.mean()) / returns.std())
                extreme_moves = z_scores > 5
                scores[extreme_moves] -= 0.15
        
        return np.clip(scores, 0.0, 1.0)


class DataIngestionPipeline:
    """Orchestrates data collection from multiple sources."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = get_logger("data.ingestion")
        self.validator = DataValidator()
        self.reconciler = DataReconciler(self.logger)
        self.sources = self._initialize_sources()
    
    def _initialize_sources(self) -> Dict[str, DataSource]:
        """Initialize configured data sources."""
        sources = {}
        
        for source_name in self.config.get_enabled_data_sources():
            source_config = self.config.get_data_source_config(source_name)
            
            if source_name == 'nikkei_official':
                sources[source_name] = NikkeiOfficialSource(source_name, source_config)
            elif source_name == 'investing_com':
                sources[source_name] = InvestingComSource(source_name, source_config)
            elif source_name == 'jpx_official':
                sources[source_name] = JPXOfficialSource(source_name, source_config)
            elif source_name == 'local_file':
                # Assuming local file path is specified
                file_path = self.config.data_dir / "nikkei225_historical.csv"
                sources[source_name] = LocalFileSource(source_name, source_config, str(file_path))
            
            # Validate connection
            try:
                if sources[source_name].validate_connection():
                    self.logger.info(f"Successfully connected to {source_name}")
                else:
                    self.logger.warning(f"Connection validation failed for {source_name}")
            except Exception as e:
                self.logger.error(f"Failed to validate {source_name}: {e}")
        
        return sources
    
    async def collect_data(self, 
                          start_date: datetime, 
                          end_date: datetime) -> pd.DataFrame:
        """Collect and reconcile data from all sources."""
        self.logger.info(f"Starting data collection from {start_date} to {end_date}")
        
        # Collect data from all sources concurrently
        tasks = []
        for source_name, source in self.sources.items():
            task = self._fetch_from_source(source, start_date, end_date)
            tasks.append(task)
        
        # Wait for all sources to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        source_data = {}
        for i, (source_name, source) in enumerate(self.sources.items()):
            result = results[i]
            
            if isinstance(result, Exception):
                self.logger.error(f"Failed to collect from {source_name}: {result}")
            elif isinstance(result, DataFetchResult) and result.success:
                source_data[source_name] = result.data
                self.logger.info(f"Successfully collected {len(result.data)} records from {source_name}")
            else:
                self.logger.warning(f"No data collected from {source_name}")
        
        if not source_data:
            raise RuntimeError("Failed to collect data from any source")
        
        # Reconcile data from multiple sources
        reconciled_data = self.reconciler.reconcile_sources(source_data)
        
        # Validate final dataset
        validation_results = self.validator.validate_dataset(reconciled_data)
        
        if not validation_results.is_valid:
            self.logger.warning(f"Data validation issues found: {validation_results.issues}")
        
        self.logger.info(f"Data collection complete. Final dataset: {len(reconciled_data)} records")
        
        return reconciled_data
    
    async def _fetch_from_source(self, 
                                source: DataSource, 
                                start_date: datetime, 
                                end_date: datetime) -> DataFetchResult:
        """Fetch data from a single source with error handling."""
        start_time = time.time()
        
        try:
            data = await source.fetch_data(start_date, end_date)
            latency_ms = (time.time() - start_time) * 1000
            
            return DataFetchResult(
                source_name=source.name,
                success=True,
                data=data,
                error=None,
                timestamp=datetime.now(),
                latency_ms=latency_ms
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Failed to fetch from {source.name}: {e}")
            
            return DataFetchResult(
                source_name=source.name,
                success=False,
                data=None,
                error=str(e),
                timestamp=datetime.now(),
                latency_ms=latency_ms
            )
    
    def collect_daily_data(self, date: datetime) -> pd.DataFrame:
        """Collect data for a specific date."""
        return asyncio.run(self.collect_data(date, date))
    
    def collect_historical_data(self, 
                               start_date: datetime, 
                               end_date: datetime) -> pd.DataFrame:
        """Collect historical data for a date range."""
        return asyncio.run(self.collect_data(start_date, end_date))