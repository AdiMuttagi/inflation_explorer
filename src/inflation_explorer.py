import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

SERIES = {
    "All Items": "CPIAUCSL",
    "Food & Beverages": "CUSR0000SAF11",
    "Energy": "CUSR0000SEHC",
    "Housing": "CUSR0000SEEA"
}

def fetch_cpi(series_codes, start="2000-01-01", end=None):
    frames = []
    for name, code in series_codes.items():
        df = pdr.DataReader(code, "fred", start, end)
        df = df.rename(columns={code: name})
        frames.append(df)
    return pd.concat(frames, axis=1)

def calculate_yoy(df):
    yoy = df.pct_change(periods=12) * 100
    return yoy.dropna()

def plot_inflation(yoy, title="Year-over-Year Inflation by Category"):
    plt.figure(figsize=(12, 6))
    for col in yoy.columns:
        plt.plot(yoy.index, yoy[col], label=col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Inflation Rate (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_rolling_stats(yoy, window=12):
    rolling_mean = yoy.rolling(window=window).mean()
    rolling_std = yoy.rolling(window=window).std()
    return rolling_mean, rolling_std

def event_driven(yoy, top_n=3):
    spikes = yoy["All Items"].nlargest(top_n)
    print(f"Top {top_n} inflation spikes (All Items):")
    for date, val in spikes.items():
        print(f"  {date.date()}: {val:.2f}%")

def forecast_trend(yoy, periods=12):
    series = yoy["All Items"]
    x = np.arange(len(series))
    y = series.values
    m, b = np.polyfit(x, y, 1)
    x_future = np.arange(len(series), len(series) + periods)
    y_future = m * x_future + b
    future_dates = pd.date_range(
        start=series.index[-1] + pd.DateOffset(months=1),
        periods=periods,
        freq='MS'
    )
    return pd.Series(y_future, index=future_dates)

def forecast_ar1(yoy, periods=12):
    series = yoy["All Items"]
    y1 = series.iloc[1:].values
    x1 = series.iloc[:-1].values
    a, b = np.polyfit(x1, y1, 1)
    last = series.iloc[-1]
    forecasts = []
    for _ in range(periods):
        pred = a * last + b
        forecasts.append(pred)
        last = pred
    future_dates = pd.date_range(
        start=series.index[-1] + pd.DateOffset(months=1),
        periods=periods,
        freq='MS'
    )
    return pd.Series(forecasts, index=future_dates)

def main():
    cpi = fetch_cpi(SERIES, start="2000-01-01")
    cpi.to_csv("data/cpi_categories.csv")

    yoy = calculate_yoy(cpi)
    yoy.to_csv("data/yoy_categories.csv")

    print("\nMost recent 5 year-over-year inflation rates (All Items):")
    print(yoy["All Items"].tail(), "\n")

    plot_inflation(yoy)

    # Plot 12-month rolling statistics
    rolling_mean, rolling_std = calculate_rolling_stats(yoy)
    rm_last12 = rolling_mean["All Items"].tail(12)
    rs_last12 = rolling_std["All Items"].tail(12)

    plt.figure(figsize=(10, 4))
    plt.plot(rm_last12.index, rm_last12, marker="o")
    plt.title("12-Month Rolling Mean of YoY Inflation — Last Year")
    plt.xlabel("Date")
    plt.ylabel("Mean YoY Inflation Rate (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(rs_last12.index, rs_last12, marker="o", color="orange")
    plt.title("12-Month Rolling Volatility of YoY Inflation — Last Year")
    plt.xlabel("Date")
    plt.ylabel("YoY Inflation Volatility (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    event_driven(yoy)

    # Forecasts
    trend_forecast = forecast_trend(yoy, periods=12)
    print("\nForecast for the next 12 months using linear trend model:")
    print(trend_forecast, "\n")

    ar1_forecast = forecast_ar1(yoy, periods=12)
    print("Forecast for the next 12 months using AR(1) model:")
    print(ar1_forecast, "\n")

    # Plot forecasts vs historical
    plt.figure(figsize=(12, 6))
    plt.plot(yoy.index, yoy["All Items"], label="Historical YoY")
    plt.plot(trend_forecast.index, trend_forecast, linestyle="--", label="Trend Forecast")
    plt.plot(ar1_forecast.index, ar1_forecast, linestyle=":", label="AR(1) Forecast")
    plt.title("Historical and Forecasted YoY Inflation (All Items)")
    plt.xlabel("Date")
    plt.ylabel("Inflation Rate (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run main analysis pipeline
if __name__ == "__main__":
    main()
