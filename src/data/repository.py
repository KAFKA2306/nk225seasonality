import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd


class MarketDataRepository:
    def __init__(self, config):
        self.config = config
        self.db_path = config.data_dir / "nikkei_data.db"
        self._ensure_database_schema()
        self._cache = {}
        self._cache_ttl = 300
        self._cache_timestamps = {}

    def _ensure_database_schema(self):
        schema_sql = """
        CREATE TABLE IF NOT EXISTS nikkei_daily_data (
            trade_date TEXT PRIMARY KEY, open_price REAL NOT NULL, high_price REAL NOT NULL,
            low_price REAL NOT NULL, close_price REAL NOT NULL, volume INTEGER,
            adjusted_close REAL NOT NULL, data_source TEXT NOT NULL, data_quality_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS data_validation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, trade_date TEXT NOT NULL, source_name TEXT NOT NULL,
            validation_rule TEXT NOT NULL, status TEXT NOT NULL, details TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS data_lineage (
            id INTEGER PRIMARY KEY AUTOINCREMENT, trade_date TEXT NOT NULL, operation TEXT NOT NULL,
            source_data TEXT, result_data TEXT, metadata TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_trade_date ON nikkei_daily_data(trade_date);
        """
        with self._get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        yield conn
        conn.close()

    def store_data(self, data: pd.DataFrame, source: str = "unknown") -> bool:
        with self._get_connection() as conn:
            for idx, row in data.iterrows():
                trade_date = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)
                record = {
                    "trade_date": trade_date,
                    "open_price": float(row.get("open_price", 0)),
                    "high_price": float(row.get("high_price", 0)),
                    "low_price": float(row.get("low_price", 0)),
                    "close_price": float(row.get("close_price", 0)),
                    "volume": int(row.get("volume", 0)) if pd.notna(row.get("volume")) else 0,
                    "adjusted_close": float(row.get("adjusted_close", row.get("close_price", 0))),
                    "data_source": source,
                    "data_quality_score": float(row.get("data_quality_score", 1.0)),
                    "updated_at": datetime.now().isoformat(),
                }
                conn.execute(
                    """
                    INSERT OR REPLACE INTO nikkei_daily_data 
                    (trade_date, open_price, high_price, low_price, close_price, volume, adjusted_close, data_source, data_quality_score, updated_at)
                    VALUES (:trade_date, :open_price, :high_price, :low_price, :close_price, :volume, :adjusted_close, :data_source, :data_quality_score, :updated_at)
                """,
                    record,
                )
            conn.commit()
        self._invalidate_cache()
        return True

    def get_historical_data(
        self, start_date: datetime, end_date: datetime, quality_threshold: float = 0.8
    ) -> pd.DataFrame:
        cache_key = f"historical_{start_date.date()}_{end_date.date()}_{quality_threshold}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        query = "SELECT trade_date, open_price, high_price, low_price, close_price, volume, adjusted_close, data_source FROM nikkei_daily_data WHERE trade_date BETWEEN ? AND ? AND data_quality_score >= ? ORDER BY trade_date"
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=[
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    quality_threshold,
                ],
            )

        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date")
            df["returns"] = df["adjusted_close"].pct_change()
            self._cache[cache_key] = df
            self._cache_timestamps[cache_key] = datetime.now()
        return df

    def get_seasonal_subset(self, months: List[int], years: int = 20, quality_threshold: float = 0.8) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        data = self.get_historical_data(start_date, end_date, quality_threshold)
        return data[data.index.month.isin(months)] if not data.empty else data

    def get_latest_data(self, days: int = 30) -> pd.DataFrame:
        end_date = datetime.now()
        return self.get_historical_data(end_date - timedelta(days=days), end_date)

    def log_validation_result(
        self,
        trade_date: datetime,
        source_name: str,
        validation_rule: str,
        status: str,
        details: Dict[str, Any],
    ):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO data_validation_log (trade_date, source_name, validation_rule, status, details) VALUES (?, ?, ?, ?, ?)",
                (
                    trade_date.strftime("%Y-%m-%d"),
                    source_name,
                    validation_rule,
                    status,
                    json.dumps(details),
                ),
            )
            conn.commit()

    def _is_cache_valid(self, cache_key: str) -> bool:
        return (
            cache_key in self._cache
            and (datetime.now() - self._cache_timestamps.get(cache_key, datetime.min)).seconds < self._cache_ttl
        )

    def _invalidate_cache(self):
        self._cache.clear()
        self._cache_timestamps.clear()
