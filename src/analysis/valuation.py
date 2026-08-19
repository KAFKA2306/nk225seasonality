from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from ..data.snapshots import build_snapshot_manifest, write_immutable_snapshot


@dataclass
class ValuationMetrics:
    jgb_yield: float
    current_per: float
    risk_premium: float
    eps: Optional[float] = None
    price: Optional[float] = None

    @property
    def earnings_yield(self) -> float:
        if self.current_per <= 0:
            return 0.0
        return (1 / self.current_per) * 100


class ValuationAnalyzer:
    def calculate_yield_gap(self, metrics: ValuationMetrics) -> float:
        return metrics.earnings_yield - metrics.jgb_yield

    def calculate_fair_per(self, metrics: ValuationMetrics) -> float:
        discount_rate = (metrics.jgb_yield + metrics.risk_premium) / 100
        if discount_rate <= 0:
            return float("inf")
        return 1 / discount_rate

    def calculate_valuation_status(self, metrics: ValuationMetrics) -> Dict[str, Any]:
        fair_per = self.calculate_fair_per(metrics)
        yield_gap = self.calculate_yield_gap(metrics)
        divergence_pct = ((metrics.current_per - fair_per) / fair_per) * 100

        status = valuation_status(divergence_pct)
        return {
            "current_per": metrics.current_per,
            "fair_per": round(fair_per, 2),
            "jgb_yield": metrics.jgb_yield,
            "earnings_yield": round(metrics.earnings_yield, 2),
            "yield_gap": round(yield_gap, 2),
            "divergence_pct": round(divergence_pct, 2),
            "status": status,
            "metrics": metrics,
        }


def valuation_status(divergence_pct: float) -> str:
    if pd.isna(divergence_pct):
        return "Unavailable"
    if divergence_pct > 20:
        return "Significantly Overvalued"
    if divergence_pct > 10:
        return "Overvalued"
    if divergence_pct < -20:
        return "Significantly Undervalued"
    if divergence_pct < -10:
        return "Undervalued"
    return "Fairly Valued"


def run_analysis_report(jgb_yield: float, current_per: float, risk_premium: float) -> None:
    print("\n" + "=" * 50)
    print("MARKET VALUATION ANALYSIS")
    print("=" * 50)

    metrics = ValuationMetrics(jgb_yield, current_per, risk_premium)
    analyzer = ValuationAnalyzer()
    result = analyzer.calculate_valuation_status(metrics)

    print("\nINPUTS:")
    print(f"  Current PER:     {metrics.current_per:>6.2f}x")
    print(f"  JGB Yield:       {metrics.jgb_yield:>6.2f}%")
    print(f"  Risk Premium:    {metrics.risk_premium:>6.2f}%")

    print("\nANALYSIS:")
    print(f"  Earnings Yield:  {result['earnings_yield']:>6.2f}%")
    print(f"  Yield Gap:       {result['yield_gap']:>6.2f}% (Earnings Yield - Bond Yield)")
    print(f"  Fair PER:        {result['fair_per']:>6.2f}x (1 / (Bond Yield + Risk Premium))")

    print("\nCONCLUSION:")
    color = (
        "\033[92m"
        if "Undervalued" in result["status"]
        else "\033[91m"
        if "Overvalued" in result["status"]
        else "\033[93m"
    )
    reset = "\033[0m"
    print(f"  Status:          {color}{result['status']}{reset}")
    print(f"  Divergence:      {result['divergence_pct']:>+6.2f}% from Fair Value")
    print("\n" + "=" * 50)


def _utc_naive(values: Any) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values))
    if index.tz is not None:
        return index.tz_convert("UTC").tz_localize(None)
    return index


def _persist_market_snapshot(
    frame: pd.DataFrame,
    *,
    value_column: str,
    snapshot_directory: Path,
    provider: str,
    identifier: str,
    meaning: str,
    unit: str,
    requested_start: datetime,
    requested_end: datetime,
    source_url: str,
    code_commit_sha: str | None,
) -> None:
    manifest = build_snapshot_manifest(
        frame,
        value_columns=[value_column],
        provider=provider,
        identifier=identifier,
        meaning=meaning,
        unit=unit,
        observation_timezone="UTC-normalized from provider timestamps",
        requested_start=requested_start,
        requested_end=requested_end,
        source_url=source_url,
        code_commit_sha=code_commit_sha,
    )
    write_immutable_snapshot(
        snapshot_directory,
        frame,
        manifest,
        value_columns=[value_column],
    )


def fetch_nikkei_data(
    years: int,
    *,
    snapshot_directory: Path | None = None,
    code_commit_sha: str | None = None,
) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    frame = yf.Ticker("^N225").history(start=start, end=end, interval="1mo")
    if frame.empty or "Close" not in frame:
        raise RuntimeError("Failed to fetch Nikkei 225 data")
    if snapshot_directory is not None:
        snapshot = pd.DataFrame(
            {"close": pd.to_numeric(frame["Close"], errors="coerce")},
            index=_utc_naive(frame.index),
        ).dropna()
        _persist_market_snapshot(
            snapshot,
            value_column="close",
            snapshot_directory=snapshot_directory,
            provider="Yahoo Finance via yfinance",
            identifier="^N225",
            meaning="Nikkei Stock Average index level",
            unit="index points",
            requested_start=start,
            requested_end=end,
            source_url="https://finance.yahoo.com/quote/%5EN225/history/",
            code_commit_sha=code_commit_sha,
        )
    return frame


def fetch_current_jgb_yield(ticker: str) -> float:
    """Fetch the latest 10-year JGB yield in percentage points."""
    try:
        history = yf.Ticker(ticker).history(period="5d")
        if history.empty or "Close" not in history:
            raise RuntimeError(f"No data returned for JGB ticker {ticker}")
        current_yield = float(history["Close"].dropna().iloc[-1])
        if not -1.0 <= current_yield <= 10.0:
            raise RuntimeError(f"JGB yield is outside the accepted range: {current_yield}")
        return current_yield
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch JGB yield for {ticker}: {exc}") from exc


def fetch_jgb_yield_history(
    ticker: str,
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
    *,
    snapshot_directory: Path | None = None,
    code_commit_sha: str | None = None,
) -> pd.DataFrame:
    """Fetch dated JGB observations for point-in-time valuation.

    The returned index is UTC-normalized and timezone-naive. No resampling or
    backward filling is performed here; callers use a backward as-of join so a
    valuation date can only consume a yield observed on or before that date.
    """
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    history = yf.Ticker(ticker).history(
        start=start_timestamp.to_pydatetime(),
        end=(end_timestamp + pd.Timedelta(days=1)).to_pydatetime(),
        interval="1d",
    )
    if history.empty or "Close" not in history:
        raise RuntimeError(f"No historical JGB data returned for ticker {ticker}")
    result = pd.DataFrame(
        {"jgb_yield": pd.to_numeric(history["Close"], errors="coerce")},
        index=_utc_naive(history.index),
    )
    result.index.name = "jgb_observed_at"
    result = result.dropna().sort_index()
    if result.empty:
        raise RuntimeError(f"Historical JGB data is empty after validation for ticker {ticker}")
    if not result["jgb_yield"].between(-1.0, 10.0).all():
        raise RuntimeError(f"Historical JGB data contains values outside the accepted range for {ticker}")
    if snapshot_directory is not None:
        _persist_market_snapshot(
            result,
            value_column="jgb_yield",
            snapshot_directory=snapshot_directory,
            provider="Yahoo Finance via yfinance",
            identifier=ticker,
            meaning="10-year JGB yield series used by the valuation model",
            unit="percentage points",
            requested_start=start_timestamp.to_pydatetime(),
            requested_end=end_timestamp.to_pydatetime(),
            source_url=f"https://finance.yahoo.com/quote/{ticker}/history/",
            code_commit_sha=code_commit_sha,
        )
    return result


def calculate_historical_per(
    price_data: pd.DataFrame,
    eps_provider: Optional[Callable[[Any], float]] = None,
) -> pd.DataFrame:
    frame = price_data.copy()
    frame["price"] = frame["Close"]
    if eps_provider is None:
        raise ValueError("Dynamic EPS provider is required for strict valuation analysis.")
    frame["estimated_eps"] = frame.index.map(lambda value: eps_provider(value))
    frame["estimated_eps"] = pd.to_numeric(frame["estimated_eps"], errors="coerce")
    frame.loc[frame["estimated_eps"] <= 0, "estimated_eps"] = np.nan
    frame["estimated_per"] = frame["price"] / frame["estimated_eps"]
    return frame


def apply_point_in_time_valuation(
    frame: pd.DataFrame,
    jgb_history: pd.DataFrame,
    *,
    risk_premium: float,
    per_column: str,
    current_jgb_yield: float | None = None,
) -> pd.DataFrame:
    """Attach point-in-time fair value using a backward as-of join.

    A row before the first available yield remains missing. Future yields are
    never backfilled into historical rows. An optional current-rate scenario is
    emitted under separate columns and never overwrites the historical series.
    """
    if per_column not in frame:
        raise ValueError(f"PER column not found: {per_column}")
    if "jgb_yield" not in jgb_history:
        raise ValueError("JGB history must contain a jgb_yield column")

    output = frame.copy()
    valuation_dates = _utc_naive(output.index)
    left = pd.DataFrame(
        {
            "valuation_date": valuation_dates,
            "_row_order": np.arange(len(output)),
        }
    ).sort_values("valuation_date")
    right = jgb_history[["jgb_yield"]].copy()
    right.index = _utc_naive(right.index)
    right = right.sort_index().reset_index()
    observed_column = right.columns[0]
    right = right.rename(columns={observed_column: "jgb_observed_at"})

    joined = pd.merge_asof(
        left,
        right,
        left_on="valuation_date",
        right_on="jgb_observed_at",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_row_order")

    output["jgb_yield"] = joined["jgb_yield"].to_numpy()
    output["jgb_observed_at"] = joined["jgb_observed_at"].to_numpy()
    discount_rate = output["jgb_yield"] + risk_premium
    output["fair_per"] = np.where(discount_rate > 0, 100 / discount_rate, np.nan)
    output["divergence"] = (
        (pd.to_numeric(output[per_column], errors="coerce") - output["fair_per"])
        / output["fair_per"]
        * 100
    )
    output["valuation_status"] = output["divergence"].map(valuation_status)
    output["valuation_method"] = "historical_point_in_time_jgb_asof"

    if current_jgb_yield is not None:
        current_discount_rate = current_jgb_yield + risk_premium
        current_fair_per = 100 / current_discount_rate if current_discount_rate > 0 else np.nan
        output["current_rate_revaluation_jgb_yield"] = current_jgb_yield
        output["current_rate_revaluation_fair_per"] = current_fair_per
        output["current_rate_revaluation_divergence"] = (
            (pd.to_numeric(output[per_column], errors="coerce") - current_fair_per)
            / current_fair_per
            * 100
        )

    return output


def run_time_series_report(years: int) -> None:
    from ..config import SystemConfig

    config = SystemConfig()
    risk_premium = config.valuation.risk_premium
    snapshot_root = config.data_dir / "market_snapshots"

    print("\n" + "=" * 60)
    print("TIME SERIES VALUATION ANALYSIS (POINT-IN-TIME JGB)")
    print("=" * 60)
    print(f"\nFetching {years} years of Nikkei 225 and JGB data...")

    price_data = fetch_nikkei_data(
        years,
        snapshot_directory=snapshot_root / "nikkei225",
    )
    per_frame = calculate_historical_per(
        price_data,
        eps_provider=config.valuation.get_eps_for_date,
    )
    yield_history = fetch_jgb_yield_history(
        config.valuation.jgb_ticker,
        per_frame.index.min(),
        per_frame.index.max(),
        snapshot_directory=snapshot_root / "jgb10y",
    )
    current_yield = fetch_current_jgb_yield(config.valuation.jgb_ticker)
    results = apply_point_in_time_valuation(
        per_frame,
        yield_history,
        risk_premium=risk_premium,
        per_column="estimated_per",
        current_jgb_yield=current_yield,
    )

    print(f"\nRisk premium: {risk_premium}%")
    print("Historical fair PER uses the latest JGB observation available at or before each month.")
    print("Current-rate revaluation is retained in separate columns only.")
    print("\n" + "-" * 110)
    print(
        f"{'Date':<12} {'Price':>10} {'EPS':>8} {'PER':>8} {'JGB':>7} "
        f"{'JGB date':<12} {'Fair PER':>9} {'Diverg':>8} {'Status':<25}"
    )
    print("-" * 110)

    for index, row in results.iterrows():
        observed = row["jgb_observed_at"]
        observed_text = "NA" if pd.isna(observed) else pd.Timestamp(observed).strftime("%Y-%m-%d")
        jgb_text = "NA" if pd.isna(row["jgb_yield"]) else f"{row['jgb_yield']:.2f}%"
        fair_text = "NA" if pd.isna(row["fair_per"]) else f"{row['fair_per']:.2f}x"
        divergence_text = "NA" if pd.isna(row["divergence"]) else f"{row['divergence']:+.1f}%"
        print(
            f"{index.strftime('%Y-%m'):<12} {row['price']:>10,.0f} {row['estimated_eps']:>8.0f} "
            f"{row['estimated_per']:>8.1f}x {jgb_text:>7} {observed_text:<12} "
            f"{fair_text:>9} {divergence_text:>8} {row['valuation_status']:<25}"
        )

    available = results.dropna(subset=["divergence"])
    print("-" * 60)
    print(f"\nSUMMARY ({len(available)}/{len(results)} periods with point-in-time JGB evidence)")
    if not available.empty:
        print(f"  Avg PER:          {available['estimated_per'].mean():.1f}x")
        print(f"  Avg Divergence:   {available['divergence'].mean():+.1f}%")
        print(f"  Max Undervalued:  {available['divergence'].min():+.1f}%")
        print(f"  Max Overvalued:   {available['divergence'].max():+.1f}%")
    print("=" * 60)
