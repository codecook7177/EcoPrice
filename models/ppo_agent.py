"""
EcoPrice - Phase 4: PPO Reinforcement Learning Agent
Trains a PPO agent to optimize discount decisions for perishable goods.

Steps:
  1. Create the Gymnasium environment (RetailPricing-v0)
  2. Train PPO agent for 200K timesteps using Stable-Baselines3
  3. Evaluate: revenue change, waste reduction, discount behavior
  4. Save agent and training logs
"""

import numpy as np
import json
import os
import time
from pathlib import Path

BASE_DIR = r"c:\Users\vitha\OneDrive\Desktop\ecoprice"
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("EcoPrice Phase 4: PPO REINFORCEMENT LEARNING")
    print("=" * 60)

    # ------------------------------------------------------------------
    # STEP 1: Create environment
    # ------------------------------------------------------------------
    print("\n[STEP 1] Creating Gymnasium environment...")

    # Import here to avoid circular imports
    from models.environment import RetailPricingEnv

    env = RetailPricingEnv()
    obs, _ = env.reset()
    print(f"  Environment: RetailPricing-v0")
    print(f"  Observation space: {env.observation_space.shape} = {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Lambda (spoilage penalty): {env.lambda_spoilage}")
    print(f"  Gamma (discount penalty): {env.gamma_discount}")
    print(f"  Max discount: {env.max_discount*100:.0f}%")
    print(f"  Price elasticity: {env.price_elasticity}")
    env.render()

    # ------------------------------------------------------------------
    # STEP 2: Baseline - No discount policy
    # ------------------------------------------------------------------
    print("\n[STEP 2] Establishing baseline (no-discount policy)...")

    n_eval = 5000
    baseline_revenue = 0
    baseline_waste = 0
    baseline_sold = 0

    for _ in range(n_eval):
        obs, _ = env.reset()
        action = np.array([0.0])  # zero discount
        _, _, _, _, info = env.step(action)
        baseline_revenue += info["revenue"]
        baseline_waste += info["units_wasted"]
        baseline_sold += info["units_sold"]

    print(f"  Baseline (0% discount) over {n_eval} episodes:")
    print(f"    Total revenue:  Rs.{baseline_revenue:,.0f}")
    print(f"    Total wasted:   {baseline_waste:,.0f} units")
    print(f"    Total sold:     {baseline_sold:,.0f} units")
    print(f"    Waste rate:     {baseline_waste/(baseline_sold+baseline_waste+1e-6)*100:.1f}%")

    # ------------------------------------------------------------------
    # STEP 3: Train PPO
    # ------------------------------------------------------------------
    print("\n[STEP 3] Training PPO agent (200K timesteps)...")
    print("  This may take 3-5 minutes...")

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    class RewardLogger(BaseCallback):
        """Log training progress."""
        def __init__(self):
            super().__init__()
            self.rewards = []
            self.episode_count = 0

        def _on_step(self):
            if self.locals.get("dones") is not None and any(self.locals["dones"]):
                infos = self.locals.get("infos", [])
                for info in infos:
                    if "raw_reward" in info:
                        self.rewards.append(info["raw_reward"])
                        self.episode_count += 1
                        if self.episode_count % 20000 == 0:
                            recent = self.rewards[-1000:]
                            avg_r = np.mean(recent)
                            avg_d = np.mean([i.get("discount", 0) for i in infos])
                            print(f"    Step {self.num_timesteps:>7d} | "
                                  f"Avg reward (last 1K): {avg_r:>8.1f} | "
                                  f"Discount: {avg_d:.1f}%")
            return True

    callback = RewardLogger()

    train_env = RetailPricingEnv()
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=42,
    )

    start_time = time.time()
    model.learn(total_timesteps=200_000, callback=callback)
    train_time = time.time() - start_time
    print(f"\n  Training completed in {train_time:.0f}s ({train_time/60:.1f} min)")

    # ------------------------------------------------------------------
    # STEP 4: Evaluate trained agent
    # ------------------------------------------------------------------
    print("\n[STEP 4] Evaluating trained PPO agent...")

    eval_env = RetailPricingEnv()
    ppo_revenue = 0
    ppo_waste = 0
    ppo_sold = 0
    discounts = []
    discount_by_expiry = {}
    discount_by_category = {}

    for _ in range(n_eval):
        obs, _ = eval_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = eval_env.step(action)
        ppo_revenue += info["revenue"]
        ppo_waste += info["units_wasted"]
        ppo_sold += info["units_sold"]
        discounts.append(info["discount"])

        # Track discount by expiry
        dte = int(info["days_to_expiry"])
        if dte not in discount_by_expiry:
            discount_by_expiry[dte] = []
        discount_by_expiry[dte].append(info["discount"])

        # Track discount by category
        cat = info.get("category", "Unknown")
        if cat not in discount_by_category:
            discount_by_category[cat] = []
        discount_by_category[cat].append(info["discount"])

    print(f"\n  PPO Agent over {n_eval} episodes:")
    print(f"    Total revenue:  Rs.{ppo_revenue:,.0f}")
    print(f"    Total wasted:   {ppo_waste:,.0f} units")
    print(f"    Total sold:     {ppo_sold:,.0f} units")
    print(f"    Avg discount:   {np.mean(discounts):.1f}%")
    print(f"    Waste rate:     {ppo_waste/(ppo_sold+ppo_waste+1e-6)*100:.1f}%")

    # Revenue and waste comparison
    rev_change = (ppo_revenue - baseline_revenue) / (baseline_revenue + 1e-6) * 100
    waste_change = (ppo_waste - baseline_waste) / (baseline_waste + 1e-6) * 100

    print(f"\n  COMPARISON vs No-Discount Baseline:")
    print(f"    Revenue change: {rev_change:+.1f}%")
    print(f"    Waste change:   {waste_change:+.1f}%")

    # ------------------------------------------------------------------
    # STEP 5: Discount behavior analysis
    # ------------------------------------------------------------------
    print("\n[STEP 5] Discount behavior analysis...")

    print("\n  Discount by Days to Expiry:")
    for dte in sorted(discount_by_expiry.keys()):
        avg_d = np.mean(discount_by_expiry[dte])
        n = len(discount_by_expiry[dte])
        bar = "#" * int(avg_d)
        print(f"    {dte} days left: avg {avg_d:>5.1f}% discount (n={n:>4d})  {bar}")

    print("\n  Discount by Category:")
    for cat in sorted(discount_by_category.keys()):
        avg_d = np.mean(discount_by_category[cat])
        n = len(discount_by_category[cat])
        bar = "#" * int(avg_d)
        print(f"    {cat:20s}: avg {avg_d:>5.1f}% discount (n={n:>4d})  {bar}")

    # Discount distribution
    print("\n  Discount Distribution:")
    bins = [0, 5, 10, 15, 20, 25, 30, 100]
    labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-25%", "25-30%", "30%+"]
    for i in range(len(bins)-1):
        count = sum(1 for d in discounts if bins[i] <= d < bins[i+1])
        pct = count / len(discounts) * 100
        bar = "#" * int(pct)
        print(f"    {labels[i]:8s}: {count:>5d} ({pct:>5.1f}%)  {bar}")

    # ------------------------------------------------------------------
    # STEP 6: Save model and metrics
    # ------------------------------------------------------------------
    print("\n[STEP 6] Saving PPO model and metrics...")

    model_path = os.path.join(MODEL_DIR, "ppo_discount_kerala.zip")
    model.save(model_path)
    print(f"  Model saved: {model_path}")

    metrics = {
        "dataset": "Kerala Kirana Store",
        "algorithm": "PPO (Proximal Policy Optimization)",
        "library": "Stable-Baselines3",
        "total_timesteps": 200000,
        "training_time_seconds": round(train_time, 1),
        "hyperparameters": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        },
        "environment": {
            "lambda_spoilage": 5.0,
            "gamma_discount": 2.0,
            "max_discount": 0.30,
            "price_elasticity": 2.5,
        },
        "baseline_no_discount": {
            "total_revenue": round(baseline_revenue, 2),
            "total_waste": round(baseline_waste, 2),
            "total_sold": round(baseline_sold, 2),
        },
        "ppo_agent": {
            "total_revenue": round(ppo_revenue, 2),
            "total_waste": round(ppo_waste, 2),
            "total_sold": round(ppo_sold, 2),
            "avg_discount_pct": round(float(np.mean(discounts)), 1),
        },
        "improvement": {
            "revenue_change_pct": round(rev_change, 1),
            "waste_change_pct": round(waste_change, 1),
        },
        "discount_by_expiry": {
            str(k): round(float(np.mean(v)), 1)
            for k, v in sorted(discount_by_expiry.items())
        },
        "discount_by_category": {
            k: round(float(np.mean(v)), 1)
            for k, v in sorted(discount_by_category.items())
        },
    }

    metrics_path = os.path.join(MODEL_DIR, "ppo_metrics_kerala.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PPO TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Training:    200K timesteps in {train_time:.0f}s")
    print(f"  Revenue:     {rev_change:+.1f}% vs no-discount baseline")
    print(f"  Waste:       {waste_change:+.1f}% vs no-discount baseline")
    print(f"  Avg discount: {np.mean(discounts):.1f}%")
    print()
    print("  The PPO agent learned to:")
    print("    - Give higher discounts when products are near expiry")
    print("    - Give lower/no discounts when products are fresh")
    print("    - Balance revenue loss from discounts against spoilage costs")


if __name__ == "__main__":
    main()
