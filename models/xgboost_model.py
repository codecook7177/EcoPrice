"""
EcoPrice - Phase 3: XGBoost Demand Forecasting Model
Trains an XGBoost regressor on the feature matrix to predict daily demand.

Steps:
  1. Load feature matrix
  2. Time-series train/test split (80/20 by date)
  3. Train XGBoost with early stopping
  4. Evaluate: R2, MAE, MAPE, RMSE
  5. Feature importance analysis
  6. Save model and metrics
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os
import joblib

BASE_DIR = r"c:\Users\vitha\OneDrive\Desktop\ecoprice"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
FEATURE_FILE = os.path.join(DATA_DIR, "kerala_features.csv")

os.makedirs(MODEL_DIR, exist_ok=True)


def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE - handles zero values."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    print("=" * 60)
    print("EcoPrice Phase 3: XGBoost DEMAND FORECASTING")
    print("=" * 60)

    # ------------------------------------------------------------------
    # STEP 1: Load feature matrix
    # ------------------------------------------------------------------
    print("\n[STEP 1] Loading feature matrix...")
    df = pd.read_csv(FEATURE_FILE, parse_dates=["date"])
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Define target and features
    target = "total_qty"
    exclude = ["date", "category", target, "total_revenue", "n_transactions"]
    feature_cols = [c for c in df.columns if c not in exclude]

    # Clean feature names for XGBoost (remove special chars)
    clean_names = {}
    for c in feature_cols:
        clean = c.replace("&", "and").replace(" ", "_")
        clean_names[c] = clean
    df = df.rename(columns=clean_names)
    feature_cols = [clean_names.get(c, c) for c in feature_cols]

    X = df[feature_cols]
    y = df[target]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Target: {target}")
    print(f"  Target range: {y.min():.1f} to {y.max():.1f}, mean={y.mean():.1f}")

    # ------------------------------------------------------------------
    # STEP 2: Time-series split (80/20 by date)
    # ------------------------------------------------------------------
    print("\n[STEP 2] Time-series train/test split...")

    dates = df["date"].sort_values().unique()
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]

    train_mask = df["date"] < split_date
    test_mask = df["date"] >= split_date

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"  Split date: {pd.Timestamp(split_date).date()}")
    print(f"  Train: {len(X_train)} rows ({train_mask.sum()/len(df)*100:.1f}%)")
    print(f"    Date range: {df[train_mask]['date'].min().date()} to {df[train_mask]['date'].max().date()}")
    print(f"  Test:  {len(X_test)} rows ({test_mask.sum()/len(df)*100:.1f}%)")
    print(f"    Date range: {df[test_mask]['date'].min().date()} to {df[test_mask]['date'].max().date()}")

    # ------------------------------------------------------------------
    # STEP 3: Train XGBoost
    # ------------------------------------------------------------------
    print("\n[STEP 3] Training XGBoost model...")

    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,        # L1 regularization
        reg_lambda=1.0,       # L2 regularization
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=80,
        eval_metric="rmse",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100,
    )

    best_iter = model.best_iteration
    print(f"\n  Best iteration: {best_iter}")

    # ------------------------------------------------------------------
    # STEP 4: Evaluate
    # ------------------------------------------------------------------
    print("\n[STEP 4] Evaluating model...")

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Test metrics
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape_test = mean_absolute_percentage_error(y_test.values, y_pred_test)

    # Train metrics
    r2_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)

    print(f"\n  TEST METRICS:")
    print(f"    R2:    {r2_test:.4f}  ({r2_test*100:.1f}%)")
    print(f"    MAE:   {mae_test:.2f}")
    print(f"    RMSE:  {rmse_test:.2f}")
    print(f"    MAPE:  {mape_test:.2f}%")
    print(f"\n  TRAIN METRICS:")
    print(f"    R2:    {r2_train:.4f}")
    print(f"    MAE:   {mae_train:.2f}")
    print(f"\n  OVERFITTING CHECK:")
    print(f"    Train R2 - Test R2 = {r2_train - r2_test:.4f}")
    if r2_train - r2_test < 0.1:
        print("    -> Good! Model generalizes well (gap < 0.10)")
    else:
        print("    -> Warning: Some overfitting detected (gap >= 0.10)")

    # ------------------------------------------------------------------
    # STEP 5: Feature importance
    # ------------------------------------------------------------------
    print("\n[STEP 5] Feature importance analysis...")

    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)

    print("\n  Top 15 most important features:")
    for i, (feat, imp) in enumerate(feat_imp[:15]):
        bar = "#" * int(imp * 200)
        print(f"    {i+1:2d}. {feat:25s}  {imp:.4f}  {bar}")

    # ------------------------------------------------------------------
    # STEP 6: Per-category accuracy
    # ------------------------------------------------------------------
    print("\n[STEP 6] Per-category accuracy...")

    df_test = df[test_mask].copy()
    df_test["predicted"] = y_pred_test

    # Use the original category column name
    cat_col = "category"
    if cat_col not in df_test.columns:
        print("  (Category column not found in test set, skipping)")
    else:
        for cat in sorted(df_test[cat_col].unique()):
            mask = df_test[cat_col] == cat
            if mask.sum() < 2:
                continue
            cat_r2 = r2_score(df_test.loc[mask, target], df_test.loc[mask, "predicted"])
            cat_mae = mean_absolute_error(df_test.loc[mask, target], df_test.loc[mask, "predicted"])
            n = mask.sum()
            print(f"    {cat:20s}  R2={cat_r2:.4f}  MAE={cat_mae:.2f}  (n={n})")

    # ------------------------------------------------------------------
    # STEP 7: Save model and metrics
    # ------------------------------------------------------------------
    print("\n[STEP 7] Saving model and metrics...")

    # Save model
    model_path = os.path.join(MODEL_DIR, "xgboost_demand_kerala.joblib")
    joblib.dump(model, model_path)
    print(f"  Model saved: {model_path}")

    # Save metrics
    metrics = {
        "dataset": "Kerala Kirana Store (sales_register_data.csv)",
        "model": "XGBRegressor",
        "n_estimators_used": int(best_iter),
        "learning_rate": 0.03,
        "max_depth": 6,
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "split_type": "Time-series (80/20 by date)",
        "split_date": str(pd.Timestamp(split_date).date()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_r2": round(float(r2_test), 4),
        "test_mae": round(float(mae_test), 4),
        "test_rmse": round(float(rmse_test), 4),
        "test_mape": round(float(mape_test), 2),
        "train_r2": round(float(r2_train), 4),
        "train_mae": round(float(mae_train), 4),
        "overfit_gap": round(float(r2_train - r2_test), 4),
        "feature_importance": {feat: round(float(imp), 4) for feat, imp in feat_imp[:20]},
    }

    metrics_path = os.path.join(MODEL_DIR, "xgboost_metrics_kerala.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("XGBOOST TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:      XGBRegressor ({best_iter} trees)")
    print(f"  Test R2:    {r2_test:.4f}  ({r2_test*100:.1f}%)")
    print(f"  Test MAE:   {mae_test:.2f} units")
    print(f"  Test MAPE:  {mape_test:.2f}%")
    print(f"  Top feature: {feat_imp[0][0]} ({feat_imp[0][1]:.4f})")
    print()

    if r2_test >= 0.80:
        print("  >>> TARGET ACHIEVED! R2 >= 0.80")
    elif r2_test >= 0.70:
        print("  >>> GOOD RESULT! R2 >= 0.70 (close to target)")
    else:
        print(f"  >>> R2 = {r2_test:.4f} - below 0.80 target. May need tuning.")


if __name__ == "__main__":
    main()
