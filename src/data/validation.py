from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config import JapaneseMarketConstants


class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    severity: ValidationSeverity
    rule_name: str
    description: str
    affected_dates: List[datetime]
    details: Dict[str, Any]


@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[ValidationIssue]
    summary: Dict[str, Any]
    validation_timestamp: datetime


class DataValidator:
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.price_change_threshold = 0.15
        self.volume_multiplier_threshold = 10
        self.missing_data_threshold = 0.05
        self.gap_threshold_days = 7

    def validate_dataset(self, data: pd.DataFrame) -> ValidationResult:
        issues = []
        issues.extend(self._validate_completeness(data))
        issues.extend(self._validate_price_consistency(data))
        issues.extend(self._validate_price_movements(data))
        issues.extend(self._validate_volume_patterns(data))
        issues.extend(self._validate_trading_calendar(data))
        issues.extend(self._validate_statistical_properties(data))
        issues.extend(self._validate_data_types(data))

        error_count = len([i for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]])
        warning_count = len([i for i in issues if i.severity == ValidationSeverity.WARNING])
        is_valid = error_count == 0 and (not self.strict_mode or warning_count == 0)

        summary = {
            "total_records": len(data),
            "date_range": {
                "start": data.index.min() if not data.empty else None,
                "end": data.index.max() if not data.empty else None,
            },
            "issues_by_severity": {s.value: len([i for i in issues if i.severity == s]) for s in ValidationSeverity},
            "data_quality_score": self._calculate_quality_score(data, issues),
        }
        return ValidationResult(is_valid, issues, summary, datetime.now())

    def _validate_completeness(self, data: pd.DataFrame) -> List[ValidationIssue]:
        issues = []
        if data.empty:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.CRITICAL,
                    "empty_dataset",
                    "Dataset is empty",
                    [],
                    {},
                )
            )
            return issues

        for col in ["open_price", "high_price", "low_price", "close_price"]:
            if col not in data.columns:
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.CRITICAL,
                        "missing_column",
                        f"Missing {col}",
                        [],
                        {"missing_column": col},
                    )
                )
                continue

            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                pct = missing_count / len(data)
                sev = ValidationSeverity.ERROR if pct > self.missing_data_threshold else ValidationSeverity.WARNING
                issues.append(
                    ValidationIssue(
                        sev,
                        "missing_values",
                        f"{col} missing {missing_count} values",
                        data[data[col].isnull()].index.tolist(),
                        {"column": col, "count": missing_count},
                    )
                )

        if not data.empty:
            for start, end, days in self._find_date_gaps(data.index):
                if days > self.gap_threshold_days:
                    issues.append(
                        ValidationIssue(
                            ValidationSeverity.WARNING,
                            "date_gap",
                            f"Gap: {days} days",
                            [start, end],
                            {"days": days},
                        )
                    )
        return issues

    def _validate_price_consistency(self, data: pd.DataFrame) -> List[ValidationIssue]:
        issues = []
        if not all(c in data.columns for c in ["open_price", "high_price", "low_price", "close_price"]):
            return issues

        inv_high = data["high_price"] < data[["open_price", "close_price"]].max(axis=1)
        if inv_high.any():
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "invalid_high_price",
                    "High < Open/Close",
                    data[inv_high].index.tolist(),
                    {"count": inv_high.sum()},
                )
            )

        inv_low = data["low_price"] > data[["open_price", "close_price"]].min(axis=1)
        if inv_low.any():
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "invalid_low_price",
                    "Low > Open/Close",
                    data[inv_low].index.tolist(),
                    {"count": inv_low.sum()},
                )
            )

        for col in ["open_price", "high_price", "low_price", "close_price"]:
            if col in data.columns:
                inv = data[col] <= 0
                if inv.any():
                    issues.append(
                        ValidationIssue(
                            ValidationSeverity.ERROR,
                            "non_positive_price",
                            f"{col} <= 0",
                            data[inv].index.tolist(),
                            {"count": inv.sum()},
                        )
                    )
        return issues

    def _validate_price_movements(self, data: pd.DataFrame) -> List[ValidationIssue]:
        issues = []
        if "close_price" not in data.columns or len(data) < 2:
            return issues

        ret = data["close_price"].pct_change().dropna()
        ext = abs(ret) > self.price_change_threshold
        if ext.any():
            issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    "extreme_price_movement",
                    f"Found {ext.sum()} extreme moves",
                    ret[ext].index.tolist(),
                    {"count": ext.sum()},
                )
            )

        same = (data["close_price"].diff() == 0).rolling(3).sum() >= 3
        if same.any():
            issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    "consecutive_identical_prices",
                    "3+ consecutive same prices",
                    data[same].index.tolist(),
                    {"count": same.sum()},
                )
            )
        return issues

    def _validate_volume_patterns(self, data: pd.DataFrame) -> List[ValidationIssue]:
        issues = []
        if "volume" not in data.columns:
            return issues
        vol = data["volume"].dropna()
        if vol.empty:
            return issues

        zero = vol == 0
        if zero.any():
            issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    "zero_volume",
                    "Zero volume found",
                    data[zero].index.tolist(),
                    {"count": zero.sum()},
                )
            )

        med = vol.median()
        spike = vol > med * self.volume_multiplier_threshold
        if spike.any():
            issues.append(
                ValidationIssue(
                    ValidationSeverity.INFO,
                    "volume_spike",
                    "Volume spikes found",
                    data[spike].index.tolist(),
                    {"count": spike.sum()},
                )
            )
        return issues

    def _validate_trading_calendar(self, data: pd.DataFrame) -> List[ValidationIssue]:
        issues = []
        if data.empty:
            return issues

        wk = data[data.index.dayofweek >= 5]
        if not wk.empty:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    "weekend_trading_data",
                    "Weekend data found",
                    wk.index.tolist(),
                    {"count": len(wk)},
                )
            )

        gw = data[
            (data.index.month == JapaneseMarketConstants.GOLDEN_WEEK_START[0])
            & (data.index.day >= JapaneseMarketConstants.GOLDEN_WEEK_START[1])
            & (data.index.day <= JapaneseMarketConstants.GOLDEN_WEEK_END[1])
        ]
        if not gw.empty:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.INFO,
                    "potential_holiday_data",
                    "Golden Week data",
                    gw.index.tolist(),
                    {"count": len(gw)},
                )
            )
        return issues

    def _validate_statistical_properties(self, data: pd.DataFrame) -> List[ValidationIssue]:
        issues = []
        if "close_price" not in data.columns or len(data) < 30:
            return issues

        ret = data["close_price"].pct_change().dropna()
        if len(ret) < 30:
            return issues

        std = ret.std()
        if std > 0.1:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    "high_volatility",
                    f"High volatility: {std:.2%}",
                    [],
                    {"volatility": std},
                )
            )

        from scipy import stats

        skew = stats.skew(ret)
        kurt = stats.kurtosis(ret)

        if abs(skew) > 2:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.INFO,
                    "extreme_skewness",
                    f"Skewness: {skew:.2f}",
                    [],
                    {"skewness": skew},
                )
            )
        if kurt > 10:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.INFO,
                    "extreme_kurtosis",
                    f"Kurtosis: {kurt:.2f}",
                    [],
                    {"kurtosis": kurt},
                )
            )
        return issues

    def _validate_data_types(self, data: pd.DataFrame) -> List[ValidationIssue]:
        issues = []
        for col in ["open_price", "high_price", "low_price", "close_price", "volume"]:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.ERROR,
                        "invalid_data_type",
                        f"{col} not numeric",
                        [],
                        {"column": col},
                    )
                )

        if not isinstance(data.index, pd.DatetimeIndex):
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "invalid_index_type",
                    "Index not DatetimeIndex",
                    [],
                    {},
                )
            )
        return issues

    def _find_date_gaps(self, idx: pd.DatetimeIndex) -> List[Tuple[datetime, datetime, int]]:
        if len(idx) < 2:
            return []
        gaps = []
        dates = idx.sort_values()
        for i in range(len(dates) - 1):
            diff = (dates[i + 1] - dates[i]).days
            biz = diff - (diff // 7) * 2
            if biz > self.gap_threshold_days:
                gaps.append((dates[i], dates[i + 1], biz))
        return gaps

    def _calculate_quality_score(self, data: pd.DataFrame, issues: List[ValidationIssue]) -> float:
        if data.empty:
            return 0.0
        score = 100.0
        penalties = {
            ValidationSeverity.INFO: 1,
            ValidationSeverity.WARNING: 3,
            ValidationSeverity.ERROR: 10,
            ValidationSeverity.CRITICAL: 25,
        }
        for i in issues:
            p = penalties[i.severity]
            if i.affected_dates:
                p *= min(len(i.affected_dates) / len(data), 1.0)
            score -= p
        return max(score, 0.0)

    def validate_single_record(
        self, record: pd.Series, previous_record: Optional[pd.Series] = None
    ) -> ValidationResult:
        issues = []
        df = pd.DataFrame([record]).T if isinstance(record, pd.Series) else pd.DataFrame([record])
        issues.extend(self._validate_price_consistency(df))
        issues.extend(self._validate_data_types(df))

        if previous_record is not None and "close_price" in record.index and "close_price" in previous_record.index:
            chg = (record["close_price"] - previous_record["close_price"]) / previous_record["close_price"]
            if abs(chg) > self.price_change_threshold:
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.WARNING,
                        "extreme_price_movement",
                        f"Change: {chg:.1%}",
                        [],
                        {"change": chg},
                    )
                )

        is_valid = not any(i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for i in issues)
        return ValidationResult(is_valid, issues, {"record_count": 1}, datetime.now())
