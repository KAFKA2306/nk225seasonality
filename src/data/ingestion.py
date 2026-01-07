import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import numpy as np
import pandas as pd
import yfinance as yf

from ..config import DataSourceConfig, get_logger
from .validation import DataValidator


@dataclass
class DataFetchResult:
    source_name: str
    success: bool
    data: Optional[pd.DataFrame]
    error: Optional[str]
    timestamp: datetime
    latency_ms: float


class DataSource(ABC):
    def __init__(self, name: str, config: DataSourceConfig):
        self.name = name
        self.config = config
        self.logger = get_logger(f"data.source.{name}")
        self._rate_limiter = RateLimiter(config.rate_limit)

    @abstractmethod
    async def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        pass

    async def _make_request(self, session: aiohttp.ClientSession, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        await self._rate_limiter.acquire()
        async with session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            response.raise_for_status()


class NikkeiOfficialSource(DataSource):
    async def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        url = f"{self.config.base_url}historical/daily"
        params = {
            "symbol": "NK225",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "api_key": self.config.api_key,
        }

        async with aiohttp.ClientSession() as session:
            data = await self._make_request(session, url, params)
            df = pd.DataFrame(data["data"])
            df["trade_date"] = pd.to_datetime(df["date"])
            df["source"] = self.name
            df = df.rename(
                columns={
                    "open": "open_price",
                    "high": "high_price",
                    "low": "low_price",
                    "close": "close_price",
                    "adjusted_close": "adjusted_close",
                }
            )
            return df[
                [
                    "trade_date",
                    "open_price",
                    "high_price",
                    "low_price",
                    "close_price",
                    "volume",
                    "adjusted_close",
                    "source",
                ]
            ].set_index("trade_date")

    def validate_connection(self) -> bool:
        import requests

        response = requests.get(
            f"{self.config.base_url}quote/NK225",
            params={"api_key": self.config.api_key},
        )
        return response.status_code == 200


class InvestingComSource(DataSource):
    async def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        url = f"{self.config.base_url}historical"
        params = {
            "symbol": "indices/nikkei-225",
            "resolution": "daily",
            "from": int(start_date.timestamp()),
            "to": int(end_date.timestamp()),
        }
        headers = {
            "X-API-KEY": self.config.api_key,
            "User-Agent": "Nikkei-Analysis/1.0",
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            data = await self._make_request(session, url, params)
            df = pd.DataFrame(
                {
                    "trade_date": pd.to_datetime(data["t"], unit="s"),
                    "open_price": data["o"],
                    "high_price": data["h"],
                    "low_price": data["l"],
                    "close_price": data["c"],
                    "volume": data["v"],
                    "source": self.name,
                }
            )
            df["adjusted_close"] = df["close_price"]
            return df.set_index("trade_date")

    def validate_connection(self) -> bool:
        import requests

        response = requests.get(
            f"{self.config.base_url}quote",
            params={"symbol": "indices/nikkei-225"},
            headers={"X-API-KEY": self.config.api_key},
        )
        return response.status_code == 200


class JPXOfficialSource(DataSource):
    async def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        url = f"{self.config.base_url}market-data/indices/daily"
        params = {
            "index_code": "NK225",
            "date_from": start_date.strftime("%Y%m%d"),
            "date_to": end_date.strftime("%Y%m%d"),
        }

        async with aiohttp.ClientSession() as session:
            data = await self._make_request(session, url, params)
            records = []
            for item in data["indices"]:
                if item["index_code"] == "NK225":
                    records.append(
                        {
                            "trade_date": pd.to_datetime(item["date"], format="%Y%m%d"),
                            "open_price": float(item["open_value"]),
                            "high_price": float(item["high_value"]),
                            "low_price": float(item["low_value"]),
                            "close_price": float(item["close_value"]),
                            "volume": int(item.get("volume", 0)),
                            "adjusted_close": float(item["close_value"]),
                            "source": self.name,
                        }
                    )
            df = pd.DataFrame(records)
            return df.set_index("trade_date") if not df.empty else pd.DataFrame()

    def validate_connection(self) -> bool:
        import requests

        response = requests.get(f"{self.config.base_url}market-data/indices/current")
        return response.status_code == 200


class LocalFileSource(DataSource):
    def __init__(self, name: str, config: DataSourceConfig, file_path: str):
        super().__init__(name, config)
        self.file_path = Path(file_path)

    async def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        df = pd.read_csv(self.file_path)
        df["trade_date"] = pd.to_datetime(df["Date"])
        df["source"] = self.name
        column_mapping = {
            "Open": "open_price",
            "High": "high_price",
            "Low": "low_price",
            "Close": "close_price",
            "Adj Close": "adjusted_close",
            "Volume": "volume",
        }
        df = df.rename(columns=column_mapping).set_index("trade_date")
        return df[(df.index >= start_date) & (df.index <= end_date)]

    def validate_connection(self) -> bool:
        return self.file_path.exists()


class YahooFinanceSource(DataSource):
    async def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        ticker = yf.Ticker("^N225")
        df = ticker.history(start=start_date, end=end_date)
        if df.empty:
            return pd.DataFrame()
        df = df.rename(
            columns={
                "Open": "open_price",
                "High": "high_price",
                "Low": "low_price",
                "Close": "close_price",
                "Volume": "volume",
            }
        )
        df["adjusted_close"] = df["close_price"]
        df["source"] = self.name
        df.index.name = "trade_date"
        return df[
            [
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "volume",
                "adjusted_close",
                "source",
            ]
        ]

    def validate_connection(self) -> bool:
        return True


class RateLimiter:
    def __init__(self, requests_per_hour: int):
        self.requests_per_second = requests_per_hour / 3600
        self.min_interval = 1.0 / self.requests_per_second
        self.last_request_time = 0

    async def acquire(self):
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)
        self.last_request_time = time.time()


class DataReconciler:
    def reconcile_sources(self, source_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not source_data:
            return pd.DataFrame()
        sorted_sources = list(source_data.keys())
        primary_source = sorted_sources[0]
        reconciled = source_data[primary_source].copy()

        for source_name in sorted_sources[1:]:
            source_df = source_data[source_name]
            missing_dates = source_df.index.difference(reconciled.index)
            if len(missing_dates) > 0:
                reconciled = pd.concat([reconciled, source_df.loc[missing_dates]])

        reconciled["data_quality_score"] = self._calculate_quality_score(reconciled)
        return reconciled.sort_index()

    def _calculate_quality_score(self, df: pd.DataFrame) -> pd.Series:
        scores = pd.Series(1.0, index=df.index)
        scores -= df.isnull().sum(axis=1) * 0.1
        price_cols = ["open_price", "high_price", "low_price", "close_price"]
        if all(col in df.columns for col in price_cols):
            invalid_high = df["high_price"] < df[["open_price", "close_price"]].max(axis=1)
            invalid_low = df["low_price"] > df[["open_price", "close_price"]].min(axis=1)
            scores[invalid_high | invalid_low] -= 0.2
        return np.clip(scores, 0.0, 1.0)


class DataIngestionPipeline:
    def __init__(self, config):
        self.config = config
        self.validator = DataValidator()
        self.reconciler = DataReconciler()
        self.sources = self._initialize_sources()

    def _initialize_sources(self) -> Dict[str, DataSource]:
        sources = {}
        for source_name in self.config.get_enabled_data_sources():
            source_config = self.config.get_data_source_config(source_name)
            if source_name == "nikkei_official":
                sources[source_name] = NikkeiOfficialSource(source_name, source_config)
            elif source_name == "investing_com":
                sources[source_name] = InvestingComSource(source_name, source_config)
            elif source_name == "jpx_official":
                sources[source_name] = JPXOfficialSource(source_name, source_config)
            elif source_name == "local_file":
                file_path = self.config.data_dir / "nikkei225_historical.csv"
                sources[source_name] = LocalFileSource(source_name, source_config, str(file_path))
            elif source_name == "yahoo_finance":
                sources[source_name] = YahooFinanceSource(source_name, source_config)
        return sources

    async def collect_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        tasks = []
        for source in self.sources.values():
            tasks.append(self._fetch_from_source(source, start_date, end_date))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        source_data = {}
        for i, result in enumerate(results):
            if isinstance(result, DataFetchResult) and result.success:
                source_data[result.source_name] = result.data

        if not source_data:
            raise RuntimeError("Failed to collect data from any source")

        return self.reconciler.reconcile_sources(source_data)

    async def _fetch_from_source(self, source: DataSource, start_date: datetime, end_date: datetime) -> DataFetchResult:
        start_time = time.time()
        try:
            data = await source.fetch_data(start_date, end_date)
            return DataFetchResult(
                source.name,
                True,
                data,
                None,
                datetime.now(),
                (time.time() - start_time) * 1000,
            )
        except Exception as e:
            return DataFetchResult(
                source.name,
                False,
                None,
                str(e),
                datetime.now(),
                (time.time() - start_time) * 1000,
            )
