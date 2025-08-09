"""
Data validation module for ensuring data quality and detecting anomalies.

This module provides comprehensive validation capabilities for financial data,
including statistical tests, missing data detection, and anomaly identification.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

from ..config import get_logger, JapaneseMarketConstants


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a data validation issue."""
    
    severity: ValidationSeverity
    rule_name: str
    description: str
    affected_dates: List[datetime]
    details: Dict[str, Any]


@dataclass 
class ValidationResult:
    """Result of data validation process."""
    
    is_valid: bool
    issues: List[ValidationIssue]
    summary: Dict[str, Any]
    validation_timestamp: datetime


class DataValidator:
    """Comprehensive data validation for financial market data."""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the validator.
        
        Args:
            strict_mode: If True, warnings are treated as errors
        """
        self.strict_mode = strict_mode
        self.logger = get_logger("data.validation")
        
        # Define validation rules and thresholds
        self.price_change_threshold = 0.15  # 15% daily change threshold
        self.volume_multiplier_threshold = 10  # Volume spike threshold
        self.missing_data_threshold = 0.05   # 5% missing data threshold
        self.gap_threshold_days = 7          # Maximum gap in trading days
        
    def validate_dataset(self, data: pd.DataFrame) -> ValidationResult:
        """
        Comprehensive validation of the entire dataset.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            ValidationResult with all issues found
        """
        self.logger.info(f"Starting validation of dataset with {len(data)} records")
        
        issues = []
        
        # Run all validation rules
        issues.extend(self._validate_completeness(data))
        issues.extend(self._validate_price_consistency(data))
        issues.extend(self._validate_price_movements(data))
        issues.extend(self._validate_volume_patterns(data))
        issues.extend(self._validate_trading_calendar(data))
        issues.extend(self._validate_statistical_properties(data))
        issues.extend(self._validate_data_types(data))
        
        # Determine overall validity
        error_count = len([issue for issue in issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]])
        warning_count = len([issue for issue in issues if issue.severity == ValidationSeverity.WARNING])
        
        is_valid = error_count == 0 and (not self.strict_mode or warning_count == 0)
        
        # Create summary
        summary = {
            'total_records': len(data),
            'date_range': {
                'start': data.index.min() if not data.empty else None,
                'end': data.index.max() if not data.empty else None
            },
            'issues_by_severity': {
                severity.value: len([i for i in issues if i.severity == severity])
                for severity in ValidationSeverity
            },
            'data_quality_score': self._calculate_quality_score(data, issues)
        }
        
        self.logger.info(f"Validation complete. Found {len(issues)} issues. Valid: {is_valid}")
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            summary=summary,
            validation_timestamp=datetime.now()
        )
    
    def _validate_completeness(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate data completeness and identify missing values."""
        issues = []
        
        if data.empty:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                rule_name="empty_dataset",
                description="Dataset is completely empty",
                affected_dates=[],
                details={}
            ))
            return issues
        
        # Check for missing values in critical columns
        required_columns = ['open_price', 'high_price', 'low_price', 'close_price']
        
        for col in required_columns:
            if col not in data.columns:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    rule_name="missing_column",
                    description=f"Required column {col} is missing",
                    affected_dates=[],
                    details={'missing_column': col}
                ))
                continue
            
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(data)
                severity = ValidationSeverity.ERROR if missing_pct > self.missing_data_threshold else ValidationSeverity.WARNING
                
                missing_dates = data[data[col].isnull()].index.tolist()
                
                issues.append(ValidationIssue(
                    severity=severity,
                    rule_name="missing_values",
                    description=f"{col} has {missing_count} missing values ({missing_pct:.1%})",
                    affected_dates=missing_dates,
                    details={
                        'column': col,
                        'missing_count': missing_count,
                        'missing_percentage': missing_pct
                    }
                ))
        
        # Check for gaps in trading days
        if not data.empty:
            date_gaps = self._find_date_gaps(data.index)
            for gap_start, gap_end, gap_days in date_gaps:
                if gap_days > self.gap_threshold_days:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        rule_name="date_gap",
                        description=f"Large gap in data: {gap_days} days from {gap_start} to {gap_end}",
                        affected_dates=[gap_start, gap_end],
                        details={
                            'gap_start': gap_start,
                            'gap_end': gap_end,
                            'gap_days': gap_days
                        }
                    ))
        
        return issues
    
    def _validate_price_consistency(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate internal price consistency (High >= Low, etc.)."""
        issues = []
        
        required_cols = ['open_price', 'high_price', 'low_price', 'close_price']
        if not all(col in data.columns for col in required_cols):
            return issues
        
        # High price should be >= max(Open, Close)
        invalid_high = data['high_price'] < data[['open_price', 'close_price']].max(axis=1)
        if invalid_high.any():
            affected_dates = data[invalid_high].index.tolist()
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                rule_name="invalid_high_price",
                description=f"High price is less than open/close price for {invalid_high.sum()} records",
                affected_dates=affected_dates,
                details={'count': invalid_high.sum()}
            ))
        
        # Low price should be <= min(Open, Close)
        invalid_low = data['low_price'] > data[['open_price', 'close_price']].min(axis=1)
        if invalid_low.any():
            affected_dates = data[invalid_low].index.tolist()
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                rule_name="invalid_low_price",
                description=f"Low price is greater than open/close price for {invalid_low.sum()} records",
                affected_dates=affected_dates,
                details={'count': invalid_low.sum()}
            ))
        
        # Check for zero or negative prices
        price_cols = ['open_price', 'high_price', 'low_price', 'close_price']
        for col in price_cols:
            if col in data.columns:
                invalid_prices = data[col] <= 0
                if invalid_prices.any():
                    affected_dates = data[invalid_prices].index.tolist()
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        rule_name="non_positive_price",
                        description=f"{col} has {invalid_prices.sum()} non-positive values",
                        affected_dates=affected_dates,
                        details={'column': col, 'count': invalid_prices.sum()}
                    ))
        
        return issues
    
    def _validate_price_movements(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate price movements for extreme changes."""
        issues = []
        
        if 'close_price' not in data.columns or len(data) < 2:
            return issues
        
        # Calculate daily returns
        returns = data['close_price'].pct_change().dropna()
        
        # Check for extreme price movements
        extreme_moves = abs(returns) > self.price_change_threshold
        if extreme_moves.any():
            affected_dates = returns[extreme_moves].index.tolist()
            max_move = returns[extreme_moves].abs().max()
            
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                rule_name="extreme_price_movement",
                description=f"Found {extreme_moves.sum()} extreme price movements (>{self.price_change_threshold:.1%})",
                affected_dates=affected_dates,
                details={
                    'count': extreme_moves.sum(),
                    'threshold': self.price_change_threshold,
                    'max_movement': max_move
                }
            ))
        
        # Check for consecutive identical prices (potential data issues)
        consecutive_same = (data['close_price'].diff() == 0).rolling(3).sum() >= 3
        if consecutive_same.any():
            affected_dates = data[consecutive_same].index.tolist()
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                rule_name="consecutive_identical_prices",
                description=f"Found {consecutive_same.sum()} instances of 3+ consecutive identical prices",
                affected_dates=affected_dates,
                details={'count': consecutive_same.sum()}
            ))
        
        return issues
    
    def _validate_volume_patterns(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate volume data patterns."""
        issues = []
        
        if 'volume' not in data.columns:
            return issues
        
        volume_data = data['volume'].dropna()
        if volume_data.empty:
            return issues
        
        # Check for zero volume
        zero_volume = volume_data == 0
        if zero_volume.any():
            affected_dates = data[zero_volume].index.tolist()
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                rule_name="zero_volume",
                description=f"Found {zero_volume.sum()} days with zero volume",
                affected_dates=affected_dates,
                details={'count': zero_volume.sum()}
            ))
        
        # Check for volume spikes
        median_volume = volume_data.median()
        volume_spikes = volume_data > median_volume * self.volume_multiplier_threshold
        if volume_spikes.any():
            affected_dates = data[volume_spikes].index.tolist()
            max_spike = (volume_data[volume_spikes] / median_volume).max()
            
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                rule_name="volume_spike",
                description=f"Found {volume_spikes.sum()} volume spikes (>{self.volume_multiplier_threshold}x median)",
                affected_dates=affected_dates,
                details={
                    'count': volume_spikes.sum(),
                    'threshold_multiplier': self.volume_multiplier_threshold,
                    'max_spike_multiple': max_spike
                }
            ))
        
        return issues
    
    def _validate_trading_calendar(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate against Japanese trading calendar."""
        issues = []
        
        if data.empty:
            return issues
        
        # Check for weekend data (should not exist)
        weekend_data = data[data.index.dayofweek >= 5]  # Saturday=5, Sunday=6
        if not weekend_data.empty:
            affected_dates = weekend_data.index.tolist()
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                rule_name="weekend_trading_data",
                description=f"Found {len(weekend_data)} records on weekends",
                affected_dates=affected_dates,
                details={'count': len(weekend_data)}
            ))
        
        # Check for data during known holidays (simplified check)
        # Golden Week check (rough approximation)
        golden_week_data = data[
            (data.index.month == JapaneseMarketConstants.GOLDEN_WEEK_START[0]) &
            (data.index.day >= JapaneseMarketConstants.GOLDEN_WEEK_START[1]) &
            (data.index.day <= JapaneseMarketConstants.GOLDEN_WEEK_END[1])
        ]
        
        if not golden_week_data.empty:
            affected_dates = golden_week_data.index.tolist()
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                rule_name="potential_holiday_data",
                description=f"Found {len(golden_week_data)} records during potential Golden Week period",
                affected_dates=affected_dates,
                details={'count': len(golden_week_data)}
            ))
        
        return issues
    
    def _validate_statistical_properties(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate statistical properties of the data."""
        issues = []
        
        if 'close_price' not in data.columns or len(data) < 30:
            return issues
        
        # Calculate returns for statistical tests
        returns = data['close_price'].pct_change().dropna()
        
        if len(returns) < 30:
            return issues
        
        # Check for unrealistic return distributions
        return_std = returns.std()
        if return_std > 0.1:  # 10% daily volatility is very high
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                rule_name="high_volatility",
                description=f"Daily return volatility is very high: {return_std:.2%}",
                affected_dates=[],
                details={'volatility': return_std}
            ))
        
        # Check for excessive skewness or kurtosis
        from scipy import stats
        skewness = stats.skew(returns)
        kurtosis_val = stats.kurtosis(returns)
        
        if abs(skewness) > 2:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                rule_name="extreme_skewness",
                description=f"Returns show extreme skewness: {skewness:.2f}",
                affected_dates=[],
                details={'skewness': skewness}
            ))
        
        if kurtosis_val > 10:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                rule_name="extreme_kurtosis",
                description=f"Returns show extreme kurtosis: {kurtosis_val:.2f}",
                affected_dates=[],
                details={'kurtosis': kurtosis_val}
            ))
        
        return issues
    
    def _validate_data_types(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate data types and formats."""
        issues = []
        
        # Check for proper numeric types in price columns
        numeric_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
        
        for col in numeric_cols:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        rule_name="invalid_data_type",
                        description=f"Column {col} should be numeric but is {data[col].dtype}",
                        affected_dates=[],
                        details={'column': col, 'actual_type': str(data[col].dtype)}
                    ))
        
        # Check for proper datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                rule_name="invalid_index_type",
                description=f"Index should be DatetimeIndex but is {type(data.index)}",
                affected_dates=[],
                details={'actual_type': str(type(data.index))}
            ))
        
        return issues
    
    def _find_date_gaps(self, date_index: pd.DatetimeIndex) -> List[Tuple[datetime, datetime, int]]:
        """Find gaps in the date series."""
        if len(date_index) < 2:
            return []
        
        gaps = []
        sorted_dates = date_index.sort_values()
        
        for i in range(len(sorted_dates) - 1):
            current_date = sorted_dates[i]
            next_date = sorted_dates[i + 1]
            
            # Calculate business days between dates (approximate)
            date_diff = (next_date - current_date).days
            
            # Rough estimate: exclude weekends (2 days per week)
            business_days = date_diff - (date_diff // 7) * 2
            
            if business_days > self.gap_threshold_days:
                gaps.append((current_date, next_date, business_days))
        
        return gaps
    
    def _calculate_quality_score(self, data: pd.DataFrame, issues: List[ValidationIssue]) -> float:
        """Calculate overall data quality score (0-100)."""
        if data.empty:
            return 0.0
        
        # Start with perfect score
        score = 100.0
        
        # Deduct points based on issue severity
        severity_penalties = {
            ValidationSeverity.INFO: 1,
            ValidationSeverity.WARNING: 3,
            ValidationSeverity.ERROR: 10,
            ValidationSeverity.CRITICAL: 25
        }
        
        for issue in issues:
            penalty = severity_penalties[issue.severity]
            # Scale penalty by number of affected records
            if issue.affected_dates:
                penalty *= min(len(issue.affected_dates) / len(data), 1.0)
            score -= penalty
        
        return max(score, 0.0)
    
    def validate_single_record(self, 
                              record: pd.Series, 
                              previous_record: Optional[pd.Series] = None) -> ValidationResult:
        """Validate a single data record."""
        issues = []
        
        # Create single-row DataFrame for consistency
        df = pd.DataFrame([record]).T if isinstance(record, pd.Series) else pd.DataFrame([record])
        
        # Run subset of validation rules appropriate for single record
        issues.extend(self._validate_price_consistency(df))
        issues.extend(self._validate_data_types(df))
        
        # If previous record available, check price movement
        if previous_record is not None:
            if 'close_price' in record.index and 'close_price' in previous_record.index:
                price_change = (record['close_price'] - previous_record['close_price']) / previous_record['close_price']
                if abs(price_change) > self.price_change_threshold:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        rule_name="extreme_price_movement",
                        description=f"Large price movement: {price_change:.1%}",
                        affected_dates=[record.name] if hasattr(record, 'name') else [],
                        details={'price_change': price_change}
                    ))
        
        is_valid = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for issue in issues)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            summary={'record_count': 1, 'issues_found': len(issues)},
            validation_timestamp=datetime.now()
        )