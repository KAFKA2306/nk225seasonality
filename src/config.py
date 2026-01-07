import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DataSourceConfig:
    enabled: bool
    api_key: str
    base_url: str
    rate_limit: int
    priority: int
    timeout: int = 30
    retry_attempts: int = 3


@dataclass
class AnalysisConfig:
    significance_level: float
    minimum_observations: int
    monte_carlo_simulations: int
    confidence_intervals: List[float]
    rolling_window_size: int
    max_missing_data_pct: float


@dataclass
class RiskConfig:
    max_position_size: float
    var_confidence: float
    expected_shortfall_confidence: float
    max_leverage: float
    stress_test_scenarios: int


@dataclass
class ValuationConfig:
    assumed_eps: float
    jgb_ticker: str
    risk_premium: float
    years_for_analysis: int
    historical_eps: Dict[int, float] = None

    def __post_init__(self):
        if self.historical_eps is None:
            self.historical_eps = {
                2014: 1050,
                2015: 1200,
                2016: 1180,
                2017: 1400,
                2018: 1700,
                2019: 1650,
                2020: 1600,
                2021: 2000,
                2022: 2150,
                2023: 2250,
                2024: 2400,
                2025: 2500,
                2026: 2550,
            }

    def get_eps_for_date(self, date_obj: Any) -> float:
        year = date_obj.year
        return self.historical_eps.get(year, self.assumed_eps)


@dataclass
class DatabaseConfig:
    host: str
    port: int
    database: str
    username: str
    password: str
    connection_pool_size: int = 10
    connection_timeout: int = 30


class SystemConfig:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.output_dir = self.project_root / "outputs"
        self.log_dir = self.project_root / "logs"
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        self._setup_data_sources()
        self._setup_analysis_config()
        self._setup_risk_config()
        self._setup_valuation_config()
        self._setup_database_config()
        self._setup_logging()

    def _setup_data_sources(self) -> None:
        self.data_sources = {
            "nikkei_official": DataSourceConfig(
                enabled=os.getenv("NIKKEI_ENABLED", "True").lower() == "true",
                api_key=os.getenv("NIKKEI_API_KEY", ""),
                base_url=os.getenv("NIKKEI_API_URL", "https://indexes.nikkei.co.jp/api/"),
                rate_limit=int(os.getenv("NIKKEI_RATE_LIMIT", "100")),
                priority=1,
            ),
            "investing_com": DataSourceConfig(
                enabled=os.getenv("INVESTING_ENABLED", "True").lower() == "true",
                api_key=os.getenv("INVESTING_API_KEY", ""),
                base_url=os.getenv("INVESTING_API_URL", "https://api.investing.com/"),
                rate_limit=int(os.getenv("INVESTING_RATE_LIMIT", "500")),
                priority=2,
            ),
            "jpx_official": DataSourceConfig(
                enabled=os.getenv("JPX_ENABLED", "True").lower() == "true",
                api_key=os.getenv("JPX_API_KEY", ""),
                base_url=os.getenv("JPX_API_URL", "https://www.jpx.co.jp/api/"),
                rate_limit=int(os.getenv("JPX_RATE_LIMIT", "200")),
                priority=1,
            ),
            "yahoo_finance": DataSourceConfig(
                enabled=True,
                api_key="",
                base_url="",
                rate_limit=1000,
                priority=3,
            ),
        }

    def _setup_analysis_config(self) -> None:
        self.analysis = AnalysisConfig(
            significance_level=float(os.getenv("SIGNIFICANCE_LEVEL", "0.05")),
            minimum_observations=int(os.getenv("MIN_OBSERVATIONS", "252")),
            monte_carlo_simulations=int(os.getenv("MC_SIMULATIONS", "10000")),
            confidence_intervals=[0.90, 0.95, 0.99],
            rolling_window_size=int(os.getenv("ROLLING_WINDOW_SIZE", "252")),
            max_missing_data_pct=float(os.getenv("MAX_MISSING_DATA_PCT", "0.05")),
        )

    def _setup_risk_config(self) -> None:
        self.risk = RiskConfig(
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.02")),
            var_confidence=float(os.getenv("VAR_CONFIDENCE", "0.95")),
            expected_shortfall_confidence=float(os.getenv("ES_CONFIDENCE", "0.95")),
            max_leverage=float(os.getenv("MAX_LEVERAGE", "2.0")),
            stress_test_scenarios=int(os.getenv("STRESS_TEST_SCENARIOS", "1000")),
        )

    def _setup_valuation_config(self) -> None:
        self.valuation = ValuationConfig(
            assumed_eps=float(os.getenv("VALUATION_EPS", "2400")),
            jgb_ticker=os.getenv("VALUATION_JGB_TICKER", "^JP10Y"),
            risk_premium=float(os.getenv("VALUATION_RISK_PREMIUM", "3.5")),
            years_for_analysis=int(os.getenv("VALUATION_YEARS", "10")),
        )

    def _setup_database_config(self) -> None:
        self.database = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "nikkei_seasonality"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            connection_pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            connection_timeout=int(os.getenv("DB_TIMEOUT", "30")),
        )

    def _setup_logging(self) -> None:
        self.logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                }
            },
            "loggers": {"": {"handlers": ["console"], "level": "INFO", "propagate": False}},
        }

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.database.username}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.database}"

    def get_data_source_config(self, source_name: str) -> DataSourceConfig:
        return self.data_sources[source_name]

    def get_enabled_data_sources(self) -> List[str]:
        return [name for name, config in self.data_sources.items() if config.enabled]

    def export_config(self) -> Dict[str, Any]:
        return {
            "data_sources": {
                name: {"enabled": c.enabled, "base_url": c.base_url} for name, c in self.data_sources.items()
            },
            "analysis": {"significance_level": self.analysis.significance_level},
            "risk": {"max_position_size": self.risk.max_position_size},
            "valuation": {
                "assumed_eps": self.valuation.assumed_eps,
                "jgb_ticker": self.valuation.jgb_ticker,
                "risk_premium": self.valuation.risk_premium,
            },
            "database": {
                "host": self.database.host,
                "database": self.database.database,
            },
        }


config = SystemConfig()


def setup_logging() -> None:
    import logging.config

    logging.config.dictConfig(config.logging_config)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class JapaneseMarketConstants:
    MARKET_OPEN_HOUR = 9
    MARKET_CLOSE_HOUR = 15
    LUNCH_BREAK_START = 11.5
    LUNCH_BREAK_END = 12.5
    FISCAL_YEAR_START_MONTH = 4
    FISCAL_YEAR_END_MONTH = 3
    GOLDEN_WEEK_START = (5, 1)
    GOLDEN_WEEK_END = (5, 7)
    DEFAULT_RISK_FREE_RATE = 0.001
    BASE_CURRENCY = "JPY"
    NIKKEI_225_SYMBOL = "^N225"
    MARKET_TIMEZONE = "Asia/Tokyo"


__all__ = [
    "SystemConfig",
    "DataSourceConfig",
    "AnalysisConfig",
    "RiskConfig",
    "DatabaseConfig",
    "JapaneseMarketConstants",
    "config",
    "setup_logging",
    "get_logger",
]
