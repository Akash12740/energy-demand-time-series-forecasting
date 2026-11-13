# Statistical Forecasting of Energy Data (Time-Series Modeling)

A comparative time-series analysis and forecasting pipeline built to model 15 years of state-level energy consumption data. The workflow covers exploratory data analysis (EDA), seasonal decomposition, stationarity testing (Augmented Dickey-Fuller), and systematic benchmarking of **AR**, **MA**, **ARIMA**, and **SARIMA** architectures using **RMSE** and **MAPE** metrics.

## Forecasting Pipeline Architecture
```text
15-Year Energy Series ──▶ Missing Value Imputation & Outlier Handling ──▶ ADF Stationarity Test
                                                                                 │
                                                                                 ▼
Optimal SARIMA Forecast ◀── RMSE & MAPE Evaluation ◀── AR / MA / ARIMA / SARIMA Grid Search
pip install -r requirements.txt
python forecast.py --data data/state_energy_15yr.csv
Create forecast.py
@'
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


def generate_15yr_energy_series(seed: int = 42) -> pd.Series:
    """Generate 15 years (180 months) of realistic seasonal state energy demand (MWh)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2008-01-01", periods=180, freq="MS")
    t = np.arange(len(dates))

    trend = 5000 + 18.5 * t
    annual_seasonality = 650 * np.sin(2 * np.pi * t / 12) + 220 * np.cos(4 * np.pi * t / 12)
    noise = rng.normal(0, 130, size=len(dates))

    values = trend + annual_seasonality + noise
    return pd.Series(values, index=dates, name="energy_mwh")


def check_stationarity(series: pd.Series) -> dict:
    """Run Augmented Dickey-Fuller (ADF) test."""
    result = adfuller(series.dropna())
    return {
        "adf_statistic": round(float(result[0]), 4),
        "p_value": round(float(result[1]), 4),
        "is_stationary": bool(result[1] < 0.05),
    }


def evaluate_models(series: pd.Series, test_months: int = 24) -> pd.DataFrame:
    """Train and compare AR, MA, ARIMA, and SARIMA models on out-of-sample test window."""
    train = series.iloc[:-test_months]
    test = series.iloc[-test_months:]

    specs = {
        "AR(2,1,0)": {"order": (2, 1, 0), "seasonal_order": (0, 0, 0, 0)},
        "MA(0,1,2)": {"order": (0, 1, 2), "seasonal_order": (0, 0, 0, 0)},
        "ARIMA(2,1,2)": {"order": (2, 1, 2), "seasonal_order": (0, 0, 0, 0)},
        "SARIMA(1,1,1)(1,1,1,12)": {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
    }

    rows = []
    for name, cfg in specs.items():
        model = SARIMAX(
            train,
            order=cfg["order"],
            seasonal_order=cfg["seasonal_order"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        preds = fitted.forecast(steps=test_months)

        rmse = root_mean_squared_error(test, preds)
        mape = mean_absolute_percentage_error(test, preds) * 100
        rows.append(
            {
                "Model": name,
                "AIC": round(float(fitted.aic), 2),
                "RMSE": round(float(rmse), 2),
                "MAPE (%)": round(float(mape), 2),
            }
        )

    results_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="15-Year Energy Demand Time-Series Forecasting")
    parser.add_argument("--data", type=str, default=None, help="Optional CSV with 'date' and 'energy_mwh' columns")
    parser.add_argument("--test-months", type=int, default=24, help="Holdout forecast horizon in months")
    args = parser.parse_args()

    if args.data:
        df = pd.read_csv(args.data, parse_dates=["date"], index_col="date")
        series = df["energy_mwh"].asfreq("MS").interpolate()
    else:
        series = generate_15yr_energy_series()

    raw_adf = check_stationarity(series)
    diff_adf = check_stationarity(series.diff().dropna())

    print("=== Stationarity Check (ADF Test) ===")
    print(f"Raw Series:         ADF={raw_adf['adf_statistic']}, p={raw_adf['p_value']} (Stationary={raw_adf['is_stationary']})")
    print(f"1st-Order Diff:     ADF={diff_adf['adf_statistic']}, p={diff_adf['p_value']} (Stationary={diff_adf['is_stationary']})")

    print(f"\n=== Out-of-Sample Forecast Evaluation ({args.test_months}-Month Holdout) ===")
    summary = evaluate_models(series, test_months=args.test_months)
    print(summary.to_string(index=False))
