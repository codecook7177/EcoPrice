"""EcoPrice Retail Pricing Environment (Gymnasium) - Kerala Dataset.

A custom gym environment that simulates a perishable product on the shelf.
The PPO agent observes the product state and decides what discount to apply.

State (observation):
    [stock, days_to_expiry, predicted_demand, price, temperature,
     humidity, is_weekend, is_holiday, day_of_week, precipitation]

Action:
    Continuous discount in [0, max_discount] (default 0-0.30 i.e. 0-30%)

Reward:
    R = revenue_from_sales - lambda * spoilage_penalty - gamma * discount_penalty

    revenue_from_sales = price * (1 - discount) * units_sold
    spoilage_penalty   = price * units_wasted
    discount_penalty   = gamma * discount^2 * units_sold   (prevents wild discounts)

    where:
        units_sold   = min(stock, demand_after_discount)
        demand_boost = 1 + elasticity * discount   (discounts increase demand)
        demand_after_discount = predicted_demand * demand_boost
        units_wasted = max(0, stock - units_sold) if days_to_expiry <= 1

Episode:
    Each episode is a single pricing decision for one product snapshot.
    The env randomly samples a product state from the Kerala dataset.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pathlib import Path


# Shelf life by category (in days)
SHELF_LIFE = {
    "Dairy": 3,
    "Eggs": 7,
    "Fruits": 5,
    "Vegetables": 4,
    "Bakery": 2,
    "Meat & Fish": 1,
    "Meat_and_Fish": 1,
    "Snacks": 30,
    "Pulses & Grains": 60,
    "Pulses_and_Grains": 60,
}


class RetailPricingEnv(gym.Env):
    """Single-step retail pricing environment for perishable goods."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path=None,
        lambda_spoilage=5.0,
        gamma_discount=2.0,
        max_discount=0.30,
        price_elasticity=2.5,
        max_stock=200,
    ):
        super().__init__()

        # Load the Kerala daily feature matrix
        data_path = data_path or (
            Path(__file__).resolve().parent.parent / "data" / "kerala_daily.csv"
        )
        self.df = pd.read_csv(data_path)

        # Hyperparameters (matching the EcoPrice reward function)
        self.lambda_spoilage = lambda_spoilage
        self.gamma_discount = gamma_discount
        self.max_discount = max_discount
        self.price_elasticity = price_elasticity
        self.max_stock = max_stock

        # Observation: [stock, days_to_expiry, predicted_demand, price,
        #               temperature, humidity, is_weekend, is_holiday,
        #               day_of_week, precipitation]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 15, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([max_stock, 60, max_stock, 1000, 45, 100, 1, 1, 6, 50], dtype=np.float32),
        )

        # Action: single continuous value = discount fraction [0, max_discount]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([max_discount], dtype=np.float32),
        )

        self.state = None
        self._current_row = None
        self._current_category = None

    def _sample_state(self):
        """Sample a realistic product state from the Kerala dataset."""
        row = self.df.sample(1).iloc[0]
        self._current_row = row

        # Get category and shelf life
        category = str(row.get("category", "Vegetables"))
        self._current_category = category
        shelf_life = SHELF_LIFE.get(category, 5)

        # Simulate stock and expiry
        predicted_demand = float(row["total_qty"])
        # Stock is typically 1-2x the daily demand
        stock = np.random.uniform(0.8, 2.0) * predicted_demand
        stock = min(max(stock, 1), self.max_stock)

        # Days to expiry: random within shelf life
        days_to_expiry = np.random.randint(0, shelf_life + 1)

        price = float(row["avg_price"])
        temperature = float(row.get("temp_mean", 28.0))
        humidity = float(row.get("humidity", 75.0))
        is_weekend = float(row.get("is_weekend", 0))
        is_holiday = float(row.get("is_holiday", 0))
        day_of_week = float(row.get("day_of_week", 0))
        precipitation = float(row.get("precipitation", 0.0))

        return np.array([
            stock, days_to_expiry, predicted_demand, price,
            temperature, humidity, is_weekend, is_holiday,
            day_of_week, precipitation
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._sample_state()
        return self.state, {}

    def step(self, action):
        discount = float(np.clip(action[0], 0.0, self.max_discount))

        stock = self.state[0]
        days_to_expiry = self.state[1]
        predicted_demand = self.state[2]
        price = self.state[3]

        # Demand increases with discount (price elasticity)
        demand_boost = 1.0 + self.price_elasticity * discount
        actual_demand = predicted_demand * demand_boost

        # Add stochasticity (real world is noisy)
        noise = np.random.normal(1.0, 0.1)
        actual_demand = max(0, actual_demand * noise)

        # Units sold = min(stock, demand)
        units_sold = min(stock, actual_demand)

        # Revenue from discounted sales
        discounted_price = price * (1.0 - discount)
        revenue = discounted_price * units_sold

        # Spoilage: unsold units that expire (only if expiring today/tomorrow)
        remaining = stock - units_sold
        if days_to_expiry <= 1:
            units_wasted = remaining
        else:
            units_wasted = 0.0

        spoilage_cost = self.lambda_spoilage * price * units_wasted

        # Discount penalty (prevents agent from always giving max discount)
        discount_penalty = self.gamma_discount * (discount ** 2) * units_sold * price

        # Final reward
        reward = revenue - spoilage_cost - discount_penalty

        # Normalize reward for stable training
        reward_normalized = np.clip(reward / (price * stock + 1e-6), -5.0, 5.0)

        # Episode info for logging
        info = {
            "discount": round(discount * 100, 1),
            "units_sold": round(units_sold, 1),
            "units_wasted": round(units_wasted, 1),
            "revenue": round(revenue, 2),
            "spoilage_cost": round(spoilage_cost, 2),
            "discount_penalty": round(discount_penalty, 2),
            "raw_reward": round(reward, 2),
            "stock": round(stock, 0),
            "days_to_expiry": round(days_to_expiry, 0),
            "predicted_demand": round(predicted_demand, 1),
            "price": round(price, 2),
            "category": self._current_category,
        }

        # Single-step episode
        terminated = True
        truncated = False

        return self.state, float(reward_normalized), terminated, truncated, info

    def render(self):
        if self.state is None:
            return
        print(
            f"Stock: {self.state[0]:.0f} | "
            f"Expiry: {self.state[1]:.0f}d | "
            f"Demand: {self.state[2]:.1f} | "
            f"Price: Rs.{self.state[3]:.0f} | "
            f"Temp: {self.state[4]:.1f}C | "
            f"Weekend: {'Y' if self.state[6] else 'N'} | "
            f"Holiday: {'Y' if self.state[7] else 'N'} | "
            f"Category: {self._current_category}"
        )


# Register the environment
gym.register(
    id="RetailPricing-v0",
    entry_point="models.environment:RetailPricingEnv",
    max_episode_steps=1,
)


if __name__ == "__main__":
    env = RetailPricingEnv()
    obs, info = env.reset()
    print("Environment created successfully!")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    env.render()

    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(f"\nAction (discount): {action[0]*100:.1f}%")
    print(f"Reward: {reward:.4f}")
    print(f"Info: {info}")
    print("\nEnvironment test passed!")
