"""
EcoPrice - Phase 3: XGBoost demand forecaster

Trains and evaluates the category-level demand forecaster on
data/kerala_features.csv (built by src/feature_extraction.py).

  * chronological split: first 80% of dates for training, last 20% for test
  * the last 20% of training dates form a validation window for early stopping;
    the final model is refit on all training rows with the chosen tree count
  * split-dependent statistics (price ratio, hot-day flag, gap filling) use
    training dates only
  * two baselines that need no training: mean of the previous 7 recorded days
    and the value on the same weekday one week earlier
  * checks: five seeds, a small grid selected on validation RMSE, and the
    test window without 31 March (Eid al-Fitr)

Outputs: models/saved/xgboost_demand.joblib, models/saved/xgboost_metrics.json,
         data/forecasts_test.csv

Usage:  python -m models.xgboost_model
"""

import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "saved"
FEATURE_FILE = DATA_DIR / "kerala_features.csv"

PARAMS = dict(learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8,
              colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42,
              n_jobs=-1, eval_metric="rmse")


def metrics(y_true, y_pred):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    nz = yt != 0
    return {"r2": float(r2_score(yt, yp)), "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mape": float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100)}


def prepare(df, train_mask):
    """Add split-dependent features using training rows only."""
    df = df.copy()
    extra = []
    cat_price = df[train_mask].groupby("category")["avg_price_lag_1"].mean()
    df["price_ratio"] = df["avg_price_lag_1"] / df["category"].map(cat_price)
    extra.append("price_ratio")
    if "temp_mean" in df.columns:
        df["is_hot"] = (df["temp_mean"] > df.loc[train_mask, "temp_mean"].mean()).astype(int)
        extra.append("is_hot")
    num = df.select_dtypes("number").columns
    df[num] = df[num].fillna(df.loc[train_mask, num].median())
    return df, extra


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    feats_report = json.loads((DATA_DIR / "feature_report.json").read_text())
    df = pd.read_csv(FEATURE_FILE, parse_dates=["date"])

    dates = np.sort(df["date"].unique())
    split = dates[int(len(dates) * 0.8)]
    train = (df["date"] < split).to_numpy()
    test = ~train
    df, extra = prepare(df, train)
    FEATS = feats_report["features"] + extra
    X, y = df[FEATS].astype(float), df["total_qty"]

    tr_dates = np.sort(df.loc[train, "date"].unique())
    vsplit = tr_dates[int(len(tr_dates) * 0.8)]
    fit = train & (df["date"] < vsplit).to_numpy()
    val = train & (df["date"] >= vsplit).to_numpy()

    es = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=80, **PARAMS)
    es.fit(X[fit], y[fit], eval_set=[(X[fit], y[fit]), (X[val], y[val])], verbose=False)
    n_trees = es.best_iteration + 1
    curve = es.evals_result()

    model = xgb.XGBRegressor(n_estimators=n_trees, **PARAMS)
    model.fit(X[train], y[train])
    p_test = model.predict(X[test])

    out = {
        "n_features": len(FEATS), "features": FEATS,
        "dropped_constant_weather": feats_report.get("dropped_constant_weather", []),
        "split_date": str(pd.Timestamp(split).date()), "val_start": str(pd.Timestamp(vsplit).date()),
        "rows": {"fit": int(fit.sum()), "val": int(val.sum()), "train": int(train.sum()), "test": int(test.sum())},
        "dates": {"train": int(df.loc[train, "date"].nunique()), "val": int(df.loc[val, "date"].nunique()),
                  "test": int(df.loc[test, "date"].nunique())},
        "n_trees": int(n_trees),
        "validation": {"xgboost": metrics(y[val], es.predict(X[val])),
                       "mean7": metrics(y[val], df.loc[val, "qty_roll_7"]),
                       "same_weekday": metrics(y[val], df.loc[val, "qty_lag_7"])},
        "test": {"xgboost": metrics(y[test], p_test),
                 "mean7": metrics(y[test], df.loc[test, "qty_roll_7"]),
                 "same_weekday": metrics(y[test], df.loc[test, "qty_lag_7"])},
        "train_fit": metrics(y[train], model.predict(X[train])),
    }

    # checks on the test result
    seeds = []
    for s in range(5):
        m = xgb.XGBRegressor(n_estimators=n_trees, **{**PARAMS, "random_state": s})
        m.fit(X[train], y[train])
        seeds.append(r2_score(y[test], m.predict(X[test])))
    grid = []
    for depth, mcw, logt in itertools.product([2, 3, 4, 6], [1, 5, 10], [False, True]):
        p = {**PARAMS, "max_depth": depth, "min_child_weight": mcw}
        yt = np.log1p(y) if logt else y
        m = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=80, **p)
        m.fit(X[fit], yt[fit], eval_set=[(X[val], yt[val])], verbose=False)
        pv = m.predict(X[val])
        pv = np.expm1(pv) if logt else pv
        grid.append({"val_rmse": float(np.sqrt(mean_squared_error(y[val], pv))), "max_depth": depth,
                     "min_child_weight": mcw, "log_target": logt, "n_trees": int(m.best_iteration + 1)})
    best = min(grid, key=lambda g: g["val_rmse"])
    m = xgb.XGBRegressor(n_estimators=best["n_trees"],
                         **{**PARAMS, "max_depth": best["max_depth"], "min_child_weight": best["min_child_weight"]})
    m.fit(X[train], np.log1p(y[train]) if best["log_target"] else y[train])
    pb = m.predict(X[test])
    pb = np.expm1(pb) if best["log_target"] else pb
    no_eid = (df.loc[test, "date"] != pd.Timestamp("2025-03-31")).to_numpy()
    out["checks"] = {
        "seeds_r2": [float(v) for v in seeds], "seeds_mean": float(np.mean(seeds)), "seeds_sd": float(np.std(seeds)),
        "grid_best": best, "grid_best_test_r2": float(r2_score(y[test], pb)),
        "without_31_march": {"xgboost": float(r2_score(y[test][no_eid], p_test[no_eid])),
                             "mean7": float(r2_score(y[test][no_eid], df.loc[test, "qty_roll_7"][no_eid]))},
    }

    te = df[test].assign(pred=p_test)
    out["per_category"] = {
        c: {"days": int(len(s)), "xgboost": float(mean_absolute_error(s["total_qty"], s["pred"])),
            "mean7": float(mean_absolute_error(s["total_qty"], s["qty_roll_7"])),
            "same_weekday": float(mean_absolute_error(s["total_qty"], s["qty_lag_7"]))}
        for c, s in te.groupby("category")}
    imp = pd.Series(model.feature_importances_, index=FEATS).sort_values(ascending=False)
    out["importance"] = {k: float(v) for k, v in imp.items()}
    out["curve"] = {"fit_rmse": curve["validation_0"]["rmse"], "val_rmse": curve["validation_1"]["rmse"]}

    joblib.dump({"model": model, "features": FEATS}, MODEL_DIR / "xgboost_demand.joblib")
    (MODEL_DIR / "xgboost_metrics.json").write_text(json.dumps(out, indent=2))
    te[["date", "category", "total_qty", "pred", "qty_roll_7"]].rename(
        columns={"pred": "forecast_xgb", "qty_roll_7": "forecast_mean7"}).to_csv(DATA_DIR / "forecasts_test.csv", index=False)

    print(f"{len(FEATS)} features, {n_trees} trees")
    for w in ["validation", "test"]:
        for k, v in out[w].items():
            print(f"  {w:10s} {k:12s} R2 {v['r2']:.3f}  MAE {v['mae']:.1f}  RMSE {v['rmse']:.1f}")
    print("  checks:", json.dumps({k: v for k, v in out["checks"].items() if k != "seeds_r2"}))


if __name__ == "__main__":
    main()
