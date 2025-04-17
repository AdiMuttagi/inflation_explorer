import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

def fetch_cpi(series="CPIAUCSL", start="2000-01-01", end=None):
    """
    Download CPI series from FRED.
    Default series is 'CPIAUCSL' (All Items).
    """
    df = pdr.DataReader(series, "fred", start, end)
    df.columns = ["CPI"]
    return df

def calculate_yoy(df):
    """
    Calculate year‐over‐year inflation rate (%) from CPI.
    """
    yoy = df["CPI"].pct_change(periods=12) * 100
    return yoy.dropna().to_frame(name="YoY Inflation")

def plot_inflation(yoy):
    """
    Plot YoY inflation over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(yoy.index, yoy["YoY Inflation"], label="YoY Inflation")
    plt.title("U.S. Year‑over‑Year Inflation Rate")
    plt.xlabel("Date")
    plt.ylabel("Inflation Rate (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # 1. Fetch and save raw CPI
    cpi = fetch_cpi(start="2000-01-01")
    cpi.to_csv("data/cpi.csv")

    # 2. Compute YoY inflation
    yoy = calculate_yoy(cpi)
    yoy.to_csv("data/yoy_inflation.csv")

    # 3. Display recent data
    print("\nLatest 5 YoY inflation rates:")
    print(yoy.tail())

    # 4. Plot
    plot_inflation(yoy)

if __name__ == "__main__":
    main()
