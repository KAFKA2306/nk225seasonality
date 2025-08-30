"""Fetch PER data for Nikkei 225 and compute EPS.

This module retrieves daily PER (price-to-earnings ratio) for the Nikkei 225 index
from the Nikkei Indexes website and combines it with daily closing prices from
Yahoo Finance to compute corresponding EPS values.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import os
import asciichartpy

BASE_URL = "https://indexes.nikkei.co.jp/en/nkave/statistics/dataload"


@dataclass
class FundData:
    date: dt.date
    per: float


def fetch_per(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Fetch daily PER data between start and end dates."""
    rows: List[FundData] = []
    current = dt.date(start.year, start.month, 1)
    end_month = dt.date(end.year, end.month, 1)
    while current <= end_month:
        params = {"list": "per", "year": current.year, "month": current.month}
        res = requests.get(BASE_URL, params=params, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        for tr in soup.select("tbody tr"):
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(tds) >= 2 and tds[1]:
                date = dt.datetime.strptime(tds[0], "%b/%d/%Y").date()
                if start <= date <= end:
                    rows.append(FundData(date=date, per=float(tds[1])))
        # increment month
        if current.month == 12:
            current = dt.date(current.year + 1, 1, 1)
        else:
            current = dt.date(current.year, current.month + 1, 1)
    df = pd.DataFrame([{"date": r.date, "per": r.per} for r in rows])
    return df.sort_values("date").reset_index(drop=True)


def fetch_price(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Fetch daily closing prices from Yahoo Finance."""
    ticker = yf.Ticker("^N225")
    data = ticker.history(start=start, end=end)
    data = data.rename(columns={"Close": "close"})
    data = data[["close"]]
    data.index = data.index.date
    return data


def get_eps_per(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Return DataFrame with PER and EPS for given date range."""
    per_df = fetch_per(start, end)
    price_df = fetch_price(start, end)
    df = per_df.join(price_df, on="date", how="inner")
    df["eps"] = df["close"] / df["per"]
    return df


def ascii_plot_series(df: pd.DataFrame, col: str, path: str) -> None:
    """Generate an ASCII chart for a dataframe column and save to text file."""
    series = df[["date", col]].set_index("date")[col]
    series.index = pd.to_datetime(series.index)
    series = series.resample("ME").last()
    chart = asciichartpy.plot(series.tolist(), {"height": 10})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(chart + "\n")


def save_eps_per(start: dt.date, end: dt.date, path: str) -> pd.DataFrame:
    df = get_eps_per(start, end)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * 3)
    df = save_eps_per(start, end, "data/nk225_eps_per.csv")
    ascii_plot_series(df, "per", "charts/nk225_per_chart.txt")
    ascii_plot_series(df, "eps", "charts/nk225_eps_chart.txt")
    print(df.head())
