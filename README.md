# U.S. Inflation Data Explorer

## Description
A command-line tool to fetch, analyze, visualize and forecast US Consumer Price Index (CPI) data. Users can compare inflation across categories, compute rolling statistics, identify historical spikes, and predict future rates using two simple models.

## Features
- Fetch CPI series from FRED for All Items, Food & Beverages, Energy, Housing
- Calculate year-over-year inflation rates for each category 
- Plot category-level inflation trends 
- Compute 12‑month rolling mean and volatility for All Items  
- Identify top 3 monthly inflation spikes 
- Forecast next 12 months using linear trend and AR(1) models  
- Plot historical and forecasted inflation together 

## nstallation
1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib pandas_datareader

## Forecast Models:
- Linear trend: fits a straight line to historical inflation rates and extends it forward
- AR(1): uses last month’s rate to predict the next month iteratively

## Future Improvements
- Allow custom forecast horizon and rolling windows via command-line arguments
- Compare with exponential smoothing or ARIMA models
- Implement back testing to measure forecast accuracy
- Build an interactive dashboard with Streamlit

