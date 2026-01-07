
import yfinance as yf
import pandas as pd

def check_ticker(symbol):
    print(f"Checking {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        # Fetch 1 year to be safe, sometimes recent data is missing for bonds
        hist = ticker.history(period="1y") 
        if not hist.empty:
            print(f"SUCCESS: {symbol}")
            print(hist.tail())
            return True
        else:
            print(f"FAILED: {symbol} (Empty)")
            return False
    except Exception as e:
        print(f"FAILED: {symbol} (Error: {e})")
        return False

candidates = [
    "^JP10Y",   # Yahoo Finance standard for indices usually starts with ^
    "JP10Y=X",  # Another common yahoo format
]

for c in candidates:
    if check_ticker(c):
        break
