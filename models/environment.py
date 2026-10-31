"""EcoPrice retail pricing environment (Gymnasium), version 2.

A single-step simulation of one pricing decision for one product category
on one day, built from the daily Kerala sales table (data/kerala_daily.csv).

Observation (10 values):
    [stock, days_to_expiry, demand_signal, price, temperature,
     humidity, is_weekend, is_holiday, day_of_week, precipitation]

    demand_signal is the realized demand of the sampled day by default.
    Pass obs_demand_col to show the agent a forecast column instead
    (realized demand still drives sales).

Action:
    discount a in [0, max_discount]

Dynamics:
    demand after discount = d * (1 + elasticity * a) * eps,  eps ~ N(1, 0.1^2)
    units sold s = min(stock, demand after discount)
    waste w = stock - s if days_to_expiry <= 1 else 0

Reward:
    R = p(1-a)s - lambda * p * w - gamma * a^2 * p * s
    normalized by p * stock and clipped to [-5, 5] for training.

Changes from version 1:
    * stock is no longer capped at 200 units (the cap held stock below demand
      for most Eggs and Vegetables episodes, so they could not generate waste)
    * the environment can be restricted to a range of dates (train/test split)
    * all randomness comes from the seeded Gymnasium generator, and reset()
      accepts a fixed episode so that several policies can be compared on
      identical episodes
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

SHELF_LIFE = {
    "Dairy": 3,
    "Eggs": 7,
    "Fruits": 5,
    "Vegetables": 4,
    "Bakery": 2,
    "Meat & Fish": 1,
    "Snacks": 30,
    "Pulses & Grains": 60,
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class RetailPricingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data=None,
        date_min=None,
        date_max=None,
        obs_demand_col=None,
        lambda_spoilage=5.0,
        gamma_discount=2.0,
        max_discount=0.30,
        price_elasticity=2.5,
        stock_factor=(0.8, 2.0),
        noise_sd=0.1,
    ):
        super().__init__()
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.read_csv(data or DATA_DIR / "kerala_daily.csv")
        df["date"] = pd.to_datetime(df["date"])
        if date_min is not None:
            df = df[df["date"] >= pd.Timestamp(date_min)]
        if date_max is not None:
            df = df[df["date"] <= pd.Timestamp(date_max)]
        self.df = df.reset_index(drop=True)
        self.obs_demand_col = obs_demand_col

        self.lambda_spoilage = lambda_spoilage
        self.gamma_discount = gamma_discount
        self.max_discount = max_discount
        self.price_elasticity = price_elasticity
        self.stock_factor = stock_factor
        self.noise_sd = noise_sd

        cap = float(max(1000.0, 2.5 * self.df["total_qty"].max()))
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -10, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([cap, 60, cap, 1000, 60, 100, 1, 1, 6, 500], dtype=np.float32),
        )
        self.action_space = spaces.Box(low=np.array([0.0], dtype=np.float32),
                                       high=np.array([max_discount], dtype=np.float32))
        self.state = None
        self.episode = None

    # ------------------------------------------------------------------
    def sample_episode(self, rng):
        """Draw an episode description: data row, stock factor, days left, demand noise."""
        i = int(rng.integers(len(self.df)))
        cat = str(self.df.at[i, "category"])
        return {
            "row": i,
            "stock_factor": float(rng.uniform(*self.stock_factor)),
            "days": int(rng.integers(0, SHELF_LIFE.get(cat, 5) + 1)),
            "noise": float(rng.normal(1.0, self.noise_sd)),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ep = (options or {}).get("episode") or self.sample_episode(self.np_random)
        row = self.df.iloc[ep["row"]]
        demand = float(row["total_qty"])
        signal = float(row[self.obs_demand_col]) if self.obs_demand_col else demand
        stock = max(1.0, ep["stock_factor"] * demand)
        self.episode = dict(ep, category=str(row["category"]), demand=demand, price=float(row["avg_price"]))
        self.state = np.array([
            stock, ep["days"], max(0.0, signal), float(row["avg_price"]),
            float(row.get("temp_mean", 27.0)), float(row.get("humidity", 80.0)),
            float(row.get("is_weekend", 0)), float(row.get("is_holiday", 0)),
            float(row.get("day_of_week", 0)), float(row.get("precipitation", 0.0)),
        ], dtype=np.float32)
        return self.state, {}

    def outcome(self, discount):
        """Sales, waste and reward for a given discount on the current episode."""
        a = float(np.clip(discount, 0.0, self.max_discount))
        stock, days, price = float(self.state[0]), float(self.state[1]), self.episode["price"]
        demand = max(0.0, self.episode["demand"] * (1.0 + self.price_elasticity * a) * self.episode["noise"])
        sold = min(stock, demand)
        waste = stock - sold if days <= 1 else 0.0
        revenue = price * (1.0 - a) * sold
        reward = revenue - self.lambda_spoilage * price * waste - self.gamma_discount * a * a * price * sold
        return {
            "discount": a, "sold": sold, "waste": waste, "revenue": revenue,
            "discount_cost": price * a * sold, "raw_reward": reward,
            "stock": stock, "days": days, "price": price, "category": self.episode["category"],
        }

    def step(self, action):
        info = self.outcome(float(np.asarray(action).reshape(-1)[0]))
        r = float(np.clip(info["raw_reward"] / (info["price"] * info["stock"] + 1e-6), -5.0, 5.0))
        return self.state, r, True, False, info


gym.register(id="RetailPricing-v2", entry_point="models.environment:RetailPricingEnv", max_episode_steps=1)
