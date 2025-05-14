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

def fetch_cpi(data_dictionary, start_date="2000-01-01", end_date=None):
    all_data = []
    for category_name in data_dictionary:
        fred_code = data_dictionary[category_name]
        data = pdr.DataReader(fred_code, "fred", start_date, end_date)
        data = data.rename(columns={fred_code: category_name})
        all_data.append(data)
    result = pd.concat(all_data, axis = 1)
    return result

def calculate_yoy(df):
    yoy = df.pct_change(periods=12, fill_method=None) * 100
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

def show_biggest_inflation_spikes(yoy_data, how_many=3):
    top_spikes = yoy_data["All Items"].nlargest(how_many)
    print(f"Top {how_many} inflation months (All Items)")
    for date, value in top_spikes.items():
        print(date.strftime("%Y-%m-%d"), round(value, 2))

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

def main():
    cpi = fetch_cpi(SERIES, start_date="2000-01-01")
    cpi.to_csv("data/cpi_categories.csv")

    yoy = calculate_yoy(cpi)
    yoy.to_csv("data/yoy_categories.csv")

    recent_inflation = yoy["All Items"].tail()
    print("Most recent 5 months of inflation (All Items):")
    print(recent_inflation)
    print()

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

    show_biggest_inflation_spikes(yoy)

    # Forecasts
    trend_forecast = forecast_trend(yoy, periods=12)
    print("\nForecast for the next 12 months using linear trend model:")
    print(trend_forecast, "\n")

    # Plot forecasts vs historical
    plt.figure(figsize=(12, 6))
    plt.plot(yoy.index, yoy["All Items"], label="Historical YoY")
    plt.plot(trend_forecast.index, trend_forecast, linestyle="--", label="Trend Forecast")
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
