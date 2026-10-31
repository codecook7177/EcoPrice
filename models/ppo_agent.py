"""
EcoPrice - Phase 4: PPO pricing agent

Trains PPO agents in RetailPricing-v2 on training dates (before 14 March 2025)
and evaluates them on paired episodes against three fixed policies:

  * no discount
  * final-day rule: a fixed discount when one day or less remains, depth chosen
    on training episodes from {10%, 20%, 30%}
  * the dashboard schedule (src/markdown_schedule.py) with neutral climate (k = 1)

Every policy sees the same episodes (same row, stock, days left and demand
noise). Test episodes come from 14-31 March. In the "forecast" setting the
demand shown to the policies is the XGBoost forecast from
data/forecasts_test.csv, while realized demand still drives sales.

Usage:  python -m models.ppo_agent --seeds 0 1 2 --steps 200000
Outputs: models/saved/ppo_seed{n}.zip, models/saved/ppo_metrics.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models.environment import RetailPricingEnv, SHELF_LIFE
from src.markdown_schedule import schedule_discount

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "saved"
SPLIT_DATE = "2025-03-14"
PPO_PARAMS = dict(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
                  gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)


def load_data():
    daily = pd.read_csv(DATA_DIR / "kerala_daily.csv", parse_dates=["date"])
    fc = pd.read_csv(DATA_DIR / "forecasts_test.csv", parse_dates=["date"])
    daily = daily.merge(fc[["date", "category", "forecast_xgb"]], on=["date", "category"], how="left")
    daily["forecast_xgb"] = daily["forecast_xgb"].clip(lower=0)
    return daily


def make_env(daily, part, obs_col=None):
    if part == "train":
        return RetailPricingEnv(daily, date_max=pd.Timestamp(SPLIT_DATE) - pd.Timedelta(days=1), obs_demand_col=obs_col)
    return RetailPricingEnv(daily, date_min=SPLIT_DATE, obs_demand_col=obs_col)


def episodes(env, n, seed):
    rng = np.random.default_rng(seed)
    return [env.sample_episode(rng) for _ in range(n)]


def run(env, eps, policy):
    rows = []
    for ep in eps:
        obs, _ = env.reset(options={"episode": ep})
        rows.append(env.outcome(policy(obs, env)))
    return pd.DataFrame(rows)


def no_discount(obs, env):
    return 0.0


def final_day_rule(depth):
    return lambda obs, env: depth if obs[1] <= 1 else 0.0


def dashboard_schedule(obs, env):
    stock, days, demand = float(obs[0]), float(obs[1]), float(obs[2])
    d, _ = schedule_discount(env.episode["category"], days, int(days), demand, stock, spoilage_mult=1.0)
    return d


def ppo_policy(model):
    return lambda obs, env: float(model.predict(obs, deterministic=True)[0][0])


def summarize(r, base=None):
    s = {"revenue": r.revenue.sum(), "sold": r.sold.sum(), "waste": r.waste.sum(),
         "waste_rate": r.waste.sum() / (r.sold.sum() + r.waste.sum()),
         "mean_discount": r.discount.mean(), "discount_cost": r.discount_cost.sum(),
         "reward": r.raw_reward.sum()}
    if base is not None:
        rng = np.random.default_rng(0)
        d_rev = (r.revenue - base.revenue).to_numpy()
        d_waste = (r.waste - base.waste).to_numpy()
        idx = rng.integers(0, len(r), (2000, len(r)))
        s["revenue_change_pct"] = d_rev.sum() / base.revenue.sum() * 100
        s["revenue_change_ci95_pct"] = list(np.percentile(d_rev[idx].sum(1) / base.revenue.sum() * 100, [2.5, 97.5]))
        s["waste_change_pct"] = d_waste.sum() / max(base.waste.sum(), 1e-9) * 100
        s["extra_units_value"] = float((r.price * (r.sold - base.sold)).sum())
    return {k: (float(v) if not isinstance(v, list) else [float(x) for x in v]) for k, v in s.items()}


def profile(r):
    days = r.days.clip(upper=5).astype(int)
    bands = pd.cut(r.discount * 100, [-0.01, 5, 10, 15, 20, 25, 29.999, 30.01],
                   labels=["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30"])
    return {"by_days": {str(k): float(v) for k, v in (r.discount * 100).groupby(days).mean().items()},
            "by_category": {k: float(v) for k, v in (r.discount * 100).groupby(r.category).mean().items()},
            "bands_pct": {str(k): float(v) for k, v in (bands.value_counts(normalize=True).sort_index() * 100).items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO
    daily = load_data()

    if not args.eval_only:
        for seed in args.seeds:
            env = make_env(daily, "train")
            t0 = time.time()
            model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu", **PPO_PARAMS)
            model.learn(total_timesteps=args.steps)
            model.save(MODEL_DIR / f"ppo_seed{seed}.zip")
            n_params = sum(p.numel() for p in model.policy.parameters())
            (MODEL_DIR / f"ppo_seed{seed}_train.json").write_text(json.dumps(
                {"seconds": time.time() - t0, "steps": args.steps, "parameters": n_params}))
            print(f"seed {seed}: trained in {time.time() - t0:.0f}s, {n_params} parameters")
    if args.train_only:
        return

    models = {s: PPO.load(MODEL_DIR / f"ppo_seed{s}.zip", device="cpu") for s in args.seeds}
    out = {"seeds": args.seeds, "episodes": args.episodes,
           "train_info": {s: json.loads((MODEL_DIR / f"ppo_seed{s}_train.json").read_text()) for s in args.seeds}}

    # choose the final-day rule depth on training episodes
    tr_env = make_env(daily, "train")
    tr_eps = episodes(tr_env, args.episodes, seed=101)
    rule_scores = {d: run(tr_env, tr_eps, final_day_rule(d)).raw_reward.sum() for d in (0.10, 0.20, 0.30)}
    depth = max(rule_scores, key=rule_scores.get)
    out["rule_depth"] = depth
    out["rule_train_reward"] = {str(k): float(v) for k, v in rule_scores.items()}

    policies = {"no_discount": no_discount, "final_day_rule": final_day_rule(depth),
                "schedule": dashboard_schedule}
    policies.update({f"ppo_seed{s}": ppo_policy(m) for s, m in models.items()})

    for setting, part, obs_col in [("train_oracle", "train", None), ("test_oracle", "test", None),
                                   ("test_forecast", "test", "forecast_xgb")]:
        env = make_env(daily, part, obs_col)
        eps = episodes(env, args.episodes, seed=202 if part == "test" else 101)
        results = {name: run(env, eps, pol) for name, pol in policies.items()}
        base = results["no_discount"]
        out[setting] = {name: summarize(r, None if name == "no_discount" else base) for name, r in results.items()}
        ppo_names = [n for n in results if n.startswith("ppo")]
        out[setting]["ppo_mean"] = {k: float(np.mean([out[setting][n][k] for n in ppo_names]))
                                    for k in ["revenue_change_pct", "waste_change_pct", "waste_rate", "mean_discount"]}
        out[setting]["ppo_sd"] = {k: float(np.std([out[setting][n][k] for n in ppo_names]))
                                  for k in ["revenue_change_pct", "waste_change_pct", "waste_rate", "mean_discount"]}
        out[setting]["profiles"] = {n: profile(results[n]) for n in ppo_names + ["schedule"]}
        if setting == "test_oracle":
            b = base.assign(window=base.days <= 1, unsold=(base.stock - base.sold) > 0)
            out["category_window"] = {c: {"units_per_day": float(daily.loc[daily.category == c, "total_qty"].mean()),
                                          "window_share": 2 / (SHELF_LIFE[c] + 1),
                                          "unsold_in_window": float(g.loc[g.window, "unsold"].mean())}
                                      for c, g in b.groupby("category")}

    (MODEL_DIR / "ppo_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    for setting in ["train_oracle", "test_oracle", "test_forecast"]:
        print(f"\n{setting}")
        for name, s in out[setting].items():
            if name in ("profiles", "ppo_mean", "ppo_sd"):
                continue
            print(f"  {name:15s} waste {s['waste']:9.0f}  rate {s['waste_rate']*100:5.2f}%  "
                  f"rev chg {s.get('revenue_change_pct', 0):+6.2f}%  disc {s['mean_discount']*100:5.1f}%")


if __name__ == "__main__":
    main()
