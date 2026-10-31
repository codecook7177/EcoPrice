"""
EcoPrice - Phase 3b: XGBoost CAUSAL Model
Trains WITHOUT lag/rolling features to reveal the impact of
weather, festivals, calendar, and price on demand.

This model answers: "WHY does demand change?"

Scope note (2026 revision): this is a descriptive model. It uses same-day
inputs such as the day's average price, so it is not a forecaster and is
not used by the app or reported in the paper. In the current data the
weather columns are constant and carry no information.
The full model (Phase 3) answers: "WHAT will demand be tomorrow?"
Both models are used together in the final system.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import json
import os
import joblib

from pathlib import Path
BASE_DIR = str(Path(__file__).resolve().parent.parent)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
FEATURE_FILE = os.path.join(DATA_DIR, "kerala_daily.csv")


def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    print("=" * 60)
    print("EcoPrice Phase 3b: XGBoost CAUSAL MODEL")
    print("(No lag/rolling features - reveals weather & festival impact)")
    print("=" * 60)

    # Load data
    df = pd.read_csv(FEATURE_FILE, parse_dates=["date"])
    df = pd.concat([df, pd.get_dummies(df["category"], prefix="cat").astype(int)], axis=1)
    df["price_ratio"] = df["avg_price"] / df.groupby("category")["avg_price"].transform("mean")
    df["temp_x_weekend"] = df["temp_mean"] * df["is_weekend"]
    df["rain_x_weekend"] = df["precipitation"].fillna(0) * df["is_weekend"]
    df["holiday_x_monthstart"] = df["is_holiday"] * df["is_month_start"]
    df["is_rainy"] = (df["precipitation"].fillna(0) > 1.0).astype(int)
    df["is_hot"] = (df["temp_mean"] > df["temp_mean"].mean()).astype(int)

    target = "total_qty"

    # ---- SELECT ONLY CAUSAL FEATURES ----
    # These are features that explain WHY demand changes,
    # not features that just copy recent sales numbers.

    # Weather features
    weather_feats = ["temp_mean", "temp_max", "temp_min", "precipitation",
                     "rain", "humidity", "wind_speed", "is_rainy", "is_hot"]

    # Calendar features
    calendar_feats = ["day_of_week", "is_weekend", "day_of_month", "month",
                      "week_of_year", "is_month_start", "is_month_end", "is_holiday"]

    # Price features
    price_feats = ["avg_price", "price_ratio"]

    # Interaction features
    interaction_feats = ["temp_x_weekend", "rain_x_weekend", "holiday_x_monthstart"]

    # Category encoding
    cat_feats = [c for c in df.columns if c.startswith("cat_")]

    # Other
    other_feats = ["n_unique_products", "avg_qty_per_txn"]

    causal_features = (weather_feats + calendar_feats + price_feats +
                       interaction_feats + cat_feats + other_feats)

    # Clean column names
    clean_names = {}
    for c in df.columns:
        clean = c.replace("&", "and").replace(" ", "_")
        clean_names[c] = clean
    df = df.rename(columns=clean_names)
    causal_features = [clean_names.get(c, c) for c in causal_features]
    causal_features = [c for c in causal_features if c in df.columns]

    print(f"\n  Causal features selected: {len(causal_features)}")
    print("  Feature groups:")
    print(f"    Weather:     {len([f for f in causal_features if f in [clean_names.get(w,w) for w in weather_feats]])}")
    print(f"    Calendar:    {len([f for f in causal_features if f in [clean_names.get(w,w) for w in calendar_feats]])}")
    print(f"    Price:       {len([f for f in causal_features if f in [clean_names.get(w,w) for w in price_feats]])}")
    print(f"    Interaction: {len([f for f in causal_features if f in [clean_names.get(w,w) for w in interaction_feats]])}")
    print(f"    Category:    {len([f for f in causal_features if f.startswith('cat_')])}")
    print(f"    Other:       {len([f for f in causal_features if f in [clean_names.get(w,w) for w in other_feats]])}")
    print()
    print("  EXCLUDED: All lag features (qty_lag_*), rolling features (qty_roll_*),")
    print("            trend features (trend_*, qty_change_*), revenue lags, txn lags")
    print("  WHY: These features mask the causal effects of weather and festivals.")

    X = df[causal_features]
    y = df[target]

    # ---- TIME-SERIES SPLIT ----
    dates = df["date"].sort_values().unique()
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]

    train_mask = df["date"] < split_date
    test_mask = df["date"] >= split_date

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"\n  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"  Split date: {pd.Timestamp(split_date).date()}")

    # ---- TRAIN ----
    print("\n  Training XGBoost (causal features only)...")

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,          # shallower to prevent overfitting
        min_child_weight=10,  # more conservative
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,        # stronger L1
        reg_lambda=5.0,       # stronger L2
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100,
    )

    best_iter = model.best_iteration
    print(f"  Best iteration: {best_iter}")

    # ---- EVALUATE ----
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape_test = mean_absolute_percentage_error(y_test.values, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    print(f"\n  CAUSAL MODEL RESULTS:")
    print(f"    Test R2:    {r2_test:.4f}  ({r2_test*100:.1f}%)")
    print(f"    Test MAE:   {mae_test:.2f}")
    print(f"    Test RMSE:  {rmse_test:.2f}")
    print(f"    Test MAPE:  {mape_test:.2f}%")
    print(f"    Train R2:   {r2_train:.4f}")
    print(f"    Overfit gap: {r2_train - r2_test:.4f}")

    # ---- FEATURE IMPORTANCE ----
    print(f"\n  FEATURE IMPORTANCE (Causal Model):")
    print(f"  {'='*55}")

    importance = model.feature_importances_
    feat_imp = sorted(zip(causal_features, importance), key=lambda x: x[1], reverse=True)

    for i, (feat, imp) in enumerate(feat_imp):
        if imp < 0.005:
            continue
        bar = "#" * int(imp * 200)
        # Label the group
        if "temp" in feat or "precip" in feat or "rain" in feat or "humid" in feat or "wind" in feat or feat == "is_rainy" or feat == "is_hot":
            group = "[WEATHER]"
        elif "day_of" in feat or "is_week" in feat or "month" in feat or "week_of" in feat or "is_month" in feat or feat == "is_holiday":
            group = "[CALENDAR]"
        elif "price" in feat or "avg_price" in feat:
            group = "[PRICE]"
        elif feat.startswith("cat_"):
            group = "[CATEGORY]"
        elif "_x_" in feat:
            group = "[INTERACTION]"
        else:
            group = "[OTHER]"

        print(f"    {i+1:2d}. {feat:25s}  {imp:.4f}  {group:13s} {bar}")

    # ---- WEATHER EFFECT ANALYSIS ----
    print(f"\n  WEATHER EFFECT ANALYSIS:")
    print(f"  {'='*55}")

    # Rainy vs non-rainy days
    rainy_col = clean_names.get("is_rainy", "is_rainy")
    if rainy_col in df.columns:
        rainy_avg = df[df[rainy_col] == 1][target].mean()
        dry_avg = df[df[rainy_col] == 0][target].mean()
        n_rainy = (df[rainy_col] == 1).sum()
        print(f"    Rainy days (n={n_rainy}):     avg demand = {rainy_avg:.1f}")
        print(f"    Dry days:                     avg demand = {dry_avg:.1f}")
        if dry_avg > 0:
            print(f"    Rain impact: {((rainy_avg-dry_avg)/dry_avg)*100:+.1f}% change")

    # Hot vs cool days
    hot_col = clean_names.get("is_hot", "is_hot")
    if hot_col in df.columns:
        hot_avg = df[df[hot_col] == 1][target].mean()
        cool_avg = df[df[hot_col] == 0][target].mean()
        print(f"    Hot days:  avg demand = {hot_avg:.1f}")
        print(f"    Cool days: avg demand = {cool_avg:.1f}")
        if cool_avg > 0:
            print(f"    Heat impact: {((hot_avg-cool_avg)/cool_avg)*100:+.1f}% change")

    # ---- FESTIVAL EFFECT ANALYSIS ----
    print(f"\n  FESTIVAL EFFECT ANALYSIS:")
    print(f"  {'='*55}")

    hol_col = clean_names.get("is_holiday", "is_holiday")
    if hol_col in df.columns:
        hol_avg = df[df[hol_col] == 1][target].mean()
        normal_avg = df[df[hol_col] == 0][target].mean()
        n_hol = (df[hol_col] == 1).sum()
        print(f"    Holiday days (n={n_hol}):   avg demand = {hol_avg:.1f}")
        print(f"    Normal days:               avg demand = {normal_avg:.1f}")
        if normal_avg > 0:
            print(f"    Festival impact: {((hol_avg-normal_avg)/normal_avg)*100:+.1f}% change")

    # ---- CALENDAR EFFECT ----
    print(f"\n  DAY-OF-WEEK EFFECT:")
    print(f"  {'='*55}")
    dow_col = clean_names.get("day_of_week", "day_of_week")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for dow in range(7):
        avg = df[df[dow_col] == dow][target].mean()
        bar = "#" * int(avg / 5)
        print(f"    {day_names[dow]:12s}: {avg:>7.1f}  {bar}")

    # ---- SAVE ----
    print("\n  Saving causal model...")
    model_path = os.path.join(MODEL_DIR, "xgboost_causal_kerala.joblib")
    joblib.dump(model, model_path)

    metrics = {
        "model_type": "XGBoost Causal (no lag features)",
        "purpose": "Shows impact of weather, festivals, calendar on demand",
        "n_features": len(causal_features),
        "feature_names": causal_features,
        "test_r2": round(float(r2_test), 4),
        "test_mae": round(float(mae_test), 4),
        "test_rmse": round(float(rmse_test), 4),
        "test_mape": round(float(mape_test), 2),
        "train_r2": round(float(r2_train), 4),
        "feature_importance": {f: round(float(imp), 4) for f, imp in feat_imp},
    }

    metrics_path = os.path.join(MODEL_DIR, "xgboost_causal_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Model saved: {model_path}")
    print(f"  Metrics saved: {metrics_path}")

    # ---- SUMMARY ----
    print(f"\n  Descriptive model test R2: {r2_test:.4f}")
    print("  This model uses same-day inputs and is for interpretation only.")
    print("  Forecasting results: models/saved/xgboost_metrics.json (python -m models.xgboost_model)")

if __name__ == "__main__":
    main()
