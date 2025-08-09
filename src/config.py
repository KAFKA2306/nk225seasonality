"""
Configuration management for Nikkei 225 seasonality analysis system.

This module provides centralized configuration management for all system components,
including data sources, analysis parameters, risk management settings, and database
configuration.
"""

import os
import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataSourceConfig:
    """Configuration for external data sources."""
    
    enabled: bool
    api_key: str
    base_url: str
    rate_limit: int  # requests per hour
    priority: int
    timeout: int = 30
    retry_attempts: int = 3


@dataclass
class AnalysisConfig:
    """Configuration for statistical analysis parameters."""
    
    significance_level: float
    minimum_observations: int
    monte_carlo_simulations: int
    confidence_intervals: List[float]
    rolling_window_size: int
    max_missing_data_pct: float


@dataclass
class RiskConfig:
    """Configuration for risk management parameters."""
    
    max_position_size: float
    var_confidence: float
    expected_shortfall_confidence: float
    max_leverage: float
    stress_test_scenarios: int


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    
    host: str
    port: int
    database: str
    username: str
    password: str
    connection_pool_size: int = 10
    connection_timeout: int = 30


class SystemConfig:
    """Centralized configuration management for the entire system."""
    
    def __init__(self):
        """Initialize configuration from environment variables and defaults."""
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.output_dir = self.project_root / "outputs"
        self.log_dir = self.project_root / "logs"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        self._setup_data_sources()
        self._setup_analysis_config()
        self._setup_risk_config()
        self._setup_database_config()
        self._setup_logging()
    
    def _setup_data_sources(self) -> None:
        """Configure data source settings."""
        self.data_sources = {
            'nikkei_official': DataSourceConfig(
                enabled=os.getenv('NIKKEI_ENABLED', 'True').lower() == 'true',
                api_key=os.getenv('NIKKEI_API_KEY', ''),
                base_url=os.getenv('NIKKEI_API_URL', 'https://indexes.nikkei.co.jp/api/'),
                rate_limit=int(os.getenv('NIKKEI_RATE_LIMIT', '100')),
                priority=1,
                timeout=30,
                retry_attempts=3
            ),
            'investing_com': DataSourceConfig(
                enabled=os.getenv('INVESTING_ENABLED', 'True').lower() == 'true',
                api_key=os.getenv('INVESTING_API_KEY', ''),
                base_url=os.getenv('INVESTING_API_URL', 'https://api.investing.com/'),
                rate_limit=int(os.getenv('INVESTING_RATE_LIMIT', '500')),
                priority=2,
                timeout=30,
                retry_attempts=3
            ),
            'jpx_official': DataSourceConfig(
                enabled=os.getenv('JPX_ENABLED', 'True').lower() == 'true',
                api_key=os.getenv('JPX_API_KEY', ''),
                base_url=os.getenv('JPX_API_URL', 'https://www.jpx.co.jp/api/'),
                rate_limit=int(os.getenv('JPX_RATE_LIMIT', '200')),
                priority=1,
                timeout=30,
                retry_attempts=3
            )
        }
    
    def _setup_analysis_config(self) -> None:
        """Configure analysis parameters."""
        self.analysis = AnalysisConfig(
            significance_level=float(os.getenv('SIGNIFICANCE_LEVEL', '0.05')),
            minimum_observations=int(os.getenv('MIN_OBSERVATIONS', '252')),
            monte_carlo_simulations=int(os.getenv('MC_SIMULATIONS', '10000')),
            confidence_intervals=[0.90, 0.95, 0.99],
            rolling_window_size=int(os.getenv('ROLLING_WINDOW_SIZE', '252')),
            max_missing_data_pct=float(os.getenv('MAX_MISSING_DATA_PCT', '0.05'))
        )
    
    def _setup_risk_config(self) -> None:
        """Configure risk management parameters."""
        self.risk = RiskConfig(
            max_position_size=float(os.getenv('MAX_POSITION_SIZE', '0.02')),
            var_confidence=float(os.getenv('VAR_CONFIDENCE', '0.95')),
            expected_shortfall_confidence=float(os.getenv('ES_CONFIDENCE', '0.95')),
            max_leverage=float(os.getenv('MAX_LEVERAGE', '2.0')),
            stress_test_scenarios=int(os.getenv('STRESS_TEST_SCENARIOS', '1000'))
        )
    
    def _setup_database_config(self) -> None:
        """Configure database connection settings."""
        self.database = DatabaseConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'nikkei_seasonality'),
            username=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', ''),
            connection_pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
            connection_timeout=int(os.getenv('DB_TIMEOUT', '30'))
        )
    
    def _setup_logging(self) -> None:
        """Configure logging settings."""
        self.logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'detailed': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'standard',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'level': 'DEBUG',
                    'formatter': 'detailed',
                    'filename': str(self.log_dir / 'nikkei_analysis.log'),
                    'mode': 'a'
                },
                'error_file': {
                    'class': 'logging.FileHandler',
                    'level': 'ERROR',
                    'formatter': 'detailed',
                    'filename': str(self.log_dir / 'errors.log'),
                    'mode': 'a'
                }
            },
            'loggers': {
                '': {
                    'handlers': ['console', 'file'],
                    'level': 'DEBUG',
                    'propagate': False
                },
                'error': {
                    'handlers': ['error_file'],
                    'level': 'ERROR',
                    'propagate': False
                }
            }
        }
    
    @property
    def database_url(self) -> str:
        """Generate database connection URL."""
        return (f"postgresql://{self.database.username}:{self.database.password}"
                f"@{self.database.host}:{self.database.port}/{self.database.database}")
    
    def get_data_source_config(self, source_name: str) -> DataSourceConfig:
        """Get configuration for a specific data source."""
        if source_name not in self.data_sources:
            raise ValueError(f"Unknown data source: {source_name}")
        return self.data_sources[source_name]
    
    def get_enabled_data_sources(self) -> List[str]:
        """Get list of enabled data sources."""
        return [name for name, config in self.data_sources.items() if config.enabled]
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return any issues."""
        issues = []
        warnings = []
        
        # Validate data sources
        enabled_sources = self.get_enabled_data_sources()
        if not enabled_sources:
            issues.append("No data sources are enabled")
        
        for source_name in enabled_sources:
            config = self.data_sources[source_name]
            if not config.api_key and 'local' not in source_name:
                warnings.append(f"No API key configured for {source_name}")
        
        # Validate analysis parameters
        if self.analysis.significance_level <= 0 or self.analysis.significance_level >= 1:
            issues.append("Significance level must be between 0 and 1")
        
        if self.analysis.minimum_observations < 100:
            warnings.append("Minimum observations is very low, results may be unreliable")
        
        # Validate risk parameters
        if self.risk.max_position_size > 0.1:
            warnings.append("Maximum position size is quite high (>10%)")
        
        if self.risk.var_confidence <= 0 or self.risk.var_confidence >= 1:
            issues.append("VaR confidence must be between 0 and 1")
        
        # Validate database configuration
        if not self.database.password and self.database.host != 'localhost':
            warnings.append("No database password configured for remote host")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration as dictionary."""
        return {
            'data_sources': {
                name: {
                    'enabled': config.enabled,
                    'base_url': config.base_url,
                    'rate_limit': config.rate_limit,
                    'priority': config.priority,
                    'timeout': config.timeout,
                    'retry_attempts': config.retry_attempts
                }
                for name, config in self.data_sources.items()
            },
            'analysis': {
                'significance_level': self.analysis.significance_level,
                'minimum_observations': self.analysis.minimum_observations,
                'monte_carlo_simulations': self.analysis.monte_carlo_simulations,
                'confidence_intervals': self.analysis.confidence_intervals,
                'rolling_window_size': self.analysis.rolling_window_size,
                'max_missing_data_pct': self.analysis.max_missing_data_pct
            },
            'risk': {
                'max_position_size': self.risk.max_position_size,
                'var_confidence': self.risk.var_confidence,
                'expected_shortfall_confidence': self.risk.expected_shortfall_confidence,
                'max_leverage': self.risk.max_leverage,
                'stress_test_scenarios': self.risk.stress_test_scenarios
            },
            'database': {
                'host': self.database.host,
                'port': self.database.port,
                'database': self.database.database,
                'connection_pool_size': self.database.connection_pool_size,
                'connection_timeout': self.database.connection_timeout
            }
        }


# Global configuration instance
config = SystemConfig()


def setup_logging() -> None:
    """Setup logging configuration."""
    import logging.config
    logging.config.dictConfig(config.logging_config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


# Market-specific constants for Japanese market
class JapaneseMarketConstants:
    """Constants specific to the Japanese stock market."""
    
    # Trading hours (JST)
    MARKET_OPEN_HOUR = 9
    MARKET_CLOSE_HOUR = 15
    LUNCH_BREAK_START = 11.5  # 11:30
    LUNCH_BREAK_END = 12.5    # 12:30
    
    # Fiscal year
    FISCAL_YEAR_START_MONTH = 4  # April
    FISCAL_YEAR_END_MONTH = 3    # March
    
    # Holiday periods
    GOLDEN_WEEK_START = (5, 1)   # May 1
    GOLDEN_WEEK_END = (5, 7)     # May 7
    
    # Risk-free rate approximation (10-year JGB)
    DEFAULT_RISK_FREE_RATE = 0.001  # 0.1%
    
    # Currency
    BASE_CURRENCY = 'JPY'
    
    # Market identifiers
    NIKKEI_225_SYMBOL = '^N225'
    MARKET_TIMEZONE = 'Asia/Tokyo'


# Export commonly used items
__all__ = [
    'SystemConfig',
    'DataSourceConfig', 
    'AnalysisConfig',
    'RiskConfig',
    'DatabaseConfig',
    'JapaneseMarketConstants',
    'config',
    'setup_logging',
    'get_logger'
]