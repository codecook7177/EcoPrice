"""
EcoPrice - Phase 2: Feature Extraction

Builds the forecasting feature matrix from data/kerala_daily.csv.

For a category and date t the target is units sold on t. Every input derived
from sales, prices or transactions uses only earlier recorded days of the same
category, so the most recent value is that of day t-1. Calendar and holiday
fields refer to day t.

Weather columns are kept only if they vary. In the current kerala_daily.csv
the historical weather fetch had failed and the columns hold one constant
value, so they are dropped.

Statistics that depend on the train/test split (hot-day threshold, price ratio
to the category mean, gap filling) are computed in models/xgboost_model.py on
training dates only.

Usage:  python -m src.feature_extraction
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DAILY_FILE = DATA_DIR / "kerala_daily.csv"
FEATURE_FILE = DATA_DIR / "kerala_features.csv"
REPORT_FILE = DATA_DIR / "feature_report.json"

WEATHER_COLS = ["temp_max", "temp_min", "temp_mean", "precipitation", "rain", "humidity", "wind_speed"]
CALENDAR_COLS = ["day_of_week", "is_weekend", "day_of_month", "month", "week_of_year",
                 "is_month_start", "is_month_end", "is_holiday"]


def build_features(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["category", "date"]).reset_index(drop=True)
    g = df.groupby("category")

    # lags (previous recorded days of the same category)
    for k in [1, 2, 3, 7, 8, 14]:
        df[f"qty_lag_{k}"] = g["total_qty"].shift(k)
    df["rev_lag_1"] = g["total_revenue"].shift(1)
    df["rev_lag_7"] = g["total_revenue"].shift(7)
    df["txn_lag_1"] = g["n_transactions"].shift(1)

    # rolling statistics over earlier days
    for w in [3, 7, 14]:
        df[f"qty_roll_{w}"] = g["total_qty"].transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean())
    df["qty_std_7"] = g["total_qty"].transform(lambda x: x.shift(1).rolling(7, min_periods=2).std())
    df["qty_min_7"] = g["total_qty"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).min())
    df["qty_max_7"] = g["total_qty"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).max())

    # trend, from lagged values only
    df["trend_3_7"] = df["qty_roll_3"] / df["qty_roll_7"].replace(0, np.nan)
    df["qty_change_1"] = df["qty_lag_1"] - df["qty_lag_2"]
    df["qty_change_7"] = df["qty_lag_1"] - df["qty_lag_8"]

    # previous day's price and store activity
    df["avg_price_lag_1"] = g["avg_price"].shift(1)
    df["price_change_1"] = df["avg_price_lag_1"] - g["avg_price"].shift(2)
    df["avg_qty_per_txn_lag_1"] = g["avg_qty_per_txn"].shift(1)
    df["n_unique_products_lag_1"] = g["n_unique_products"].shift(1)

    df["holiday_x_monthstart"] = df["is_holiday"] * df["is_month_start"]

    weather = [c for c in WEATHER_COLS if c in df.columns and df[c].nunique(dropna=True) > 1]
    if weather:
        df["is_rainy"] = (df["precipitation"].fillna(0) > 1.0).astype(int)
        df["temp_x_weekend"] = df["temp_mean"] * df["is_weekend"]
        df["rain_x_weekend"] = df["precipitation"].fillna(0) * df["is_weekend"]
        weather += ["is_rainy", "temp_x_weekend", "rain_x_weekend"]

    cats = pd.get_dummies(df["category"], prefix="cat").astype(int)
    cats.columns = [c.replace("&", "and").replace(" ", "_") for c in cats.columns]
    df = pd.concat([df, cats], axis=1)

    df = df.dropna(subset=["qty_lag_7"]).reset_index(drop=True)   # 7 earlier days needed

    features = (["qty_lag_1", "qty_lag_2", "qty_lag_3", "qty_lag_7", "qty_lag_14",
                 "rev_lag_1", "rev_lag_7", "txn_lag_1",
                 "qty_roll_3", "qty_roll_7", "qty_roll_14", "qty_std_7", "qty_min_7", "qty_max_7",
                 "trend_3_7", "qty_change_1", "qty_change_7"]
                + CALENDAR_COLS + ["holiday_x_monthstart"] + weather
                + ["avg_price_lag_1", "price_change_1",
                   "avg_qty_per_txn_lag_1", "n_unique_products_lag_1"]
                + list(cats.columns))
    return df, features


def main():
    daily = pd.read_csv(DAILY_FILE)
    df, features = build_features(daily)
    dropped = [c for c in WEATHER_COLS if c in daily.columns and daily[c].nunique() <= 1]
    keep = ["date", "category"] + features + ["qty_lag_8", "total_qty"]
    df[keep].to_csv(FEATURE_FILE, index=False)
    report = {"rows": len(df), "n_features": len(features), "features": features,
              "dropped_constant_weather": dropped}
    REPORT_FILE.write_text(json.dumps(report, indent=2))
    print(f"Saved {FEATURE_FILE} ({len(df)} rows, {len(features)} features; "
          f"constant weather columns dropped: {dropped or 'none'})")


if __name__ == "__main__":
    main()
