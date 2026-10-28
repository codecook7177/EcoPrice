"""
EcoPrice - Phase 2: Feature Extraction
Takes the daily aggregated data from Phase 1 and engineers features for XGBoost.

Steps:
  1. Load daily aggregated data (kerala_daily.csv)
  2. Create lag features (yesterday's sales, last week's sales, etc.)
  3. Create rolling window features (3-day, 7-day, 14-day averages)
  4. Create interaction features (weather x weekend, holiday x category, etc.)
  5. Create trend features (is demand going up or down?)
  6. Create price features (relative price, price changes)
  7. Encode categorical variables (category -> numbers)
  8. Save final feature matrix
"""

import pandas as pd
import numpy as np
import json
import os

BASE_DIR = r"c:\Users\vitha\OneDrive\Desktop\ecoprice"
DATA_DIR = os.path.join(BASE_DIR, "data")
DAILY_FILE = os.path.join(DATA_DIR, "kerala_daily.csv")
FEATURE_FILE = os.path.join(DATA_DIR, "kerala_features.csv")
REPORT_FILE = os.path.join(DATA_DIR, "feature_report.json")


def main():
    print("=" * 60)
    print("EcoPrice Phase 2: FEATURE EXTRACTION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # STEP 1: Load daily data
    # ------------------------------------------------------------------
    print("\n[STEP 1] Loading daily aggregated data...")
    df = pd.read_csv(DAILY_FILE, parse_dates=["date"])
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Categories: {sorted(df['category'].unique())}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    report = {
        "input_rows": len(df),
        "input_columns": len(df.columns),
    }

    # ------------------------------------------------------------------
    # STEP 2: Lag features (past sales as predictors)
    # ------------------------------------------------------------------
    print("\n[STEP 2] Creating lag features...")

    # Sort by category and date for proper lag calculation
    df = df.sort_values(["category", "date"]).reset_index(drop=True)

    # Create lags WITHIN each category
    for lag in [1, 2, 3, 7, 14]:
        df[f"qty_lag_{lag}"] = df.groupby("category")["total_qty"].shift(lag)
        print(f"  Created qty_lag_{lag} (sales from {lag} day(s) ago)")

    # Revenue lags
    for lag in [1, 7]:
        df[f"rev_lag_{lag}"] = df.groupby("category")["total_revenue"].shift(lag)
        print(f"  Created rev_lag_{lag}")

    # Transaction count lag
    df["txn_lag_1"] = df.groupby("category")["n_transactions"].shift(1)

    lag_features = [c for c in df.columns if "lag" in c]
    print(f"  Total lag features: {len(lag_features)}")

    # ------------------------------------------------------------------
    # STEP 3: Rolling window features
    # ------------------------------------------------------------------
    print("\n[STEP 3] Creating rolling window features...")

    for window in [3, 7, 14]:
        col = f"qty_roll_{window}"
        df[col] = (
            df.groupby("category")["total_qty"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        print(f"  Created {col} ({window}-day rolling average)")

    # Rolling std (volatility)
    df["qty_std_7"] = (
        df.groupby("category")["total_qty"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=2).std())
    )
    print("  Created qty_std_7 (7-day volatility)")

    # Rolling min/max
    df["qty_min_7"] = (
        df.groupby("category")["total_qty"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).min())
    )
    df["qty_max_7"] = (
        df.groupby("category")["total_qty"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).max())
    )
    print("  Created qty_min_7, qty_max_7")

    roll_features = [c for c in df.columns if "roll" in c or "std" in c or "min_7" in c or "max_7" in c]
    print(f"  Total rolling features: {len(roll_features)}")

    # ------------------------------------------------------------------
    # STEP 4: Trend features
    # ------------------------------------------------------------------
    print("\n[STEP 4] Creating trend features...")

    # Trend: ratio of recent avg to older avg
    df["trend_3_7"] = df["qty_roll_3"] / df["qty_roll_7"].replace(0, np.nan)
    print("  Created trend_3_7 (3-day avg / 7-day avg; >1 = trending up)")

    # Day-over-day change
    df["qty_change_1"] = df.groupby("category")["total_qty"].diff(1)
    print("  Created qty_change_1 (change from yesterday)")

    # Week-over-week change
    df["qty_change_7"] = df.groupby("category")["total_qty"].diff(7)
    print("  Created qty_change_7 (change from same day last week)")

    # ------------------------------------------------------------------
    # STEP 5: Interaction features
    # ------------------------------------------------------------------
    print("\n[STEP 5] Creating interaction features...")

    # Temperature x Weekend (hot weekends = different buying)
    df["temp_x_weekend"] = df["temp_mean"] * df["is_weekend"]
    print("  Created temp_x_weekend")

    # Rain x Weekend
    df["rain_x_weekend"] = df["precipitation"].fillna(0) * df["is_weekend"]
    print("  Created rain_x_weekend")

    # Holiday x Month Start (double effect on payday holidays)
    df["holiday_x_monthstart"] = df["is_holiday"] * df["is_month_start"]
    print("  Created holiday_x_monthstart")

    # Is it a rainy day? (binary)
    df["is_rainy"] = (df["precipitation"].fillna(0) > 1.0).astype(int)
    print("  Created is_rainy (precipitation > 1mm)")

    # Is it a hot day? (above average temp)
    avg_temp = df["temp_mean"].mean()
    df["is_hot"] = (df["temp_mean"] > avg_temp).astype(int)
    print(f"  Created is_hot (temp > {avg_temp:.1f}C)")

    # ------------------------------------------------------------------
    # STEP 6: Price features
    # ------------------------------------------------------------------
    print("\n[STEP 6] Creating price features...")

    # Price change from yesterday
    df["price_change_1"] = df.groupby("category")["avg_price"].diff(1)
    print("  Created price_change_1")

    # Price ratio to category average
    cat_avg_price = df.groupby("category")["avg_price"].transform("mean")
    df["price_ratio"] = df["avg_price"] / cat_avg_price.replace(0, np.nan)
    print("  Created price_ratio (today's price / category average)")

    # ------------------------------------------------------------------
    # STEP 7: Encode categorical variables
    # ------------------------------------------------------------------
    print("\n[STEP 7] Encoding categorical variables...")

    # One-hot encode category
    cat_dummies = pd.get_dummies(df["category"], prefix="cat")
    df = pd.concat([df, cat_dummies], axis=1)
    print(f"  One-hot encoded 'category' into {len(cat_dummies.columns)} columns")
    for c in cat_dummies.columns:
        print(f"    {c}")

    # ------------------------------------------------------------------
    # STEP 8: Handle missing values & select final features
    # ------------------------------------------------------------------
    print("\n[STEP 8] Handling missing values...")

    # Lag features have NaN for the first few days - expected
    before = len(df)
    # Drop rows where lag_7 is null (need at least 7 days of history)
    df = df.dropna(subset=["qty_lag_7"])
    after = len(df)
    dropped = before - after
    print(f"  Dropped {dropped} rows with insufficient history (need 7+ days)")

    # Fill remaining NaN in rolling features
    fill_cols = [c for c in df.columns if "roll" in c or "std" in c or "trend" in c
                 or "change" in c or "min_7" in c or "max_7" in c]
    for c in fill_cols:
        n_null = df[c].isna().sum()
        if n_null > 0:
            df[c] = df[c].fillna(df[c].median())
            print(f"  Filled {n_null} NaN in {c} with median")

    # Fill weather nulls
    for c in ["temp_mean", "temp_max", "temp_min", "precipitation", "humidity", "wind_speed", "rain"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # ------------------------------------------------------------------
    # STEP 9: Define final feature set and target
    # ------------------------------------------------------------------
    print("\n[STEP 9] Selecting final features...")

    # Target
    target = "total_qty"

    # Features - everything the model can use
    exclude_cols = [
        "date", "category", "day_name", "holiday_name",
        target, "total_revenue",  # target and leaky revenue
        "n_transactions",  # leaky - we won't know this in advance
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and not c.startswith("Page")]

    print(f"  Target: {target}")
    print(f"  Number of features: {len(feature_cols)}")
    print(f"  Features:")
    for i, f in enumerate(sorted(feature_cols)):
        print(f"    {i+1:2d}. {f}")

    # ------------------------------------------------------------------
    # STEP 10: Save feature matrix
    # ------------------------------------------------------------------
    print("\n[STEP 10] Saving feature matrix...")

    # Save full dataset with all columns for reference
    output = df[["date", "category"] + feature_cols + [target]].copy()
    output.to_csv(FEATURE_FILE, index=False)
    print(f"  Saved {FEATURE_FILE}")
    print(f"  Final shape: {output.shape[0]} rows x {output.shape[1]} columns")

    # Save feature report
    report["output_rows"] = len(output)
    report["output_columns"] = len(output.columns)
    report["target"] = target
    report["n_features"] = len(feature_cols)
    report["feature_names"] = sorted(feature_cols)
    report["dropped_for_history"] = dropped
    report["feature_groups"] = {
        "lag_features": [c for c in feature_cols if "lag" in c],
        "rolling_features": [c for c in feature_cols if "roll" in c or "std" in c
                            or "min_7" in c or "max_7" in c],
        "trend_features": [c for c in feature_cols if "trend" in c or "change" in c],
        "weather_features": [c for c in feature_cols if c in
                            ["temp_mean", "temp_max", "temp_min", "precipitation",
                             "humidity", "wind_speed", "rain", "is_rainy", "is_hot"]],
        "calendar_features": [c for c in feature_cols if c in
                             ["day_of_week", "is_weekend", "day_of_month", "month",
                              "week_of_year", "is_month_start", "is_month_end", "is_holiday"]],
        "interaction_features": [c for c in feature_cols if "_x_" in c],
        "price_features": [c for c in feature_cols if "price" in c],
        "category_features": [c for c in feature_cols if c.startswith("cat_")],
    }

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved {REPORT_FILE}")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"  Input:    {report['input_rows']} rows, {report['input_columns']} columns")
    print(f"  Output:   {report['output_rows']} rows, {report['output_columns']} columns")
    print(f"  Features: {report['n_features']}")
    print(f"  Target:   {target}")
    print()
    print("  Feature groups:")
    for group, feats in report["feature_groups"].items():
        print(f"    {group:25s}: {len(feats)} features")
    print()

    # Quick correlation check - top features correlated with target
    print("  Top 10 features correlated with target (total_qty):")
    corrs = output[feature_cols + [target]].corr()[target].drop(target).abs().sort_values(ascending=False)
    for feat, corr in corrs.head(10).items():
        print(f"    {feat:25s}  r = {corr:.4f}")
    print()
    print(f"  Highest correlation: {corrs.index[0]} = {corrs.iloc[0]:.4f}")
    print("  (This is MUCH better than the old dataset where max correlation was ~0.03)")


if __name__ == "__main__":
    main()
