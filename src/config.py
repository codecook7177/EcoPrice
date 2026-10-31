"""EcoPrice shared config: shelf life, geo, features, APIs."""
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RAW_CANDIDATES = [
    DATA / "store_sales_data (2).csv",
    DATA / "raw.csv",
]

# 10-feature matrix matching templates/forecast.html + festival countdown
# days_to_festival: 0=festival day ... 7=7 days away, 8=no festival within 7d
FEATURE_COLS = [
    "days_to_expiry",   # int
    "temperature",      # C, Open-Meteo daily mean
    "day_of_week",      # 0=Mon..6=Sun
    "is_holiday",       # 0/1, holidays India (day-of)
    "days_to_festival", # 0..7 countdown, 8 = >7d away (ignore)
    "price",            # Sales / Quantity
    "rolling_avg_7d",   # mean Qty prior 7d per (Sub-Category, Postal Code)
    "humidity",         # %, Open-Meteo daily mean
    "precipitation",    # mm, Open-Meteo daily sum
    "month",            # 1-12
    "is_weekend",       # 0/1
]
TARGET_COL = "quantity"

# Shelf life (days) from settings.html + domain defaults.
# Dataset has no expiry metadata, so days_to_expiry is synthesized 0..shelf_life.
SHELF_LIFE = {
    "Milk": 3, "Butter": 7, "Cheese": 7, "Yogurt": 5,
    "Apples": 7, "Mangoes": 5, "Tomatoes": 4, "Carrots": 5,
    "Burgers": 2, "Pizzas": 2, "Fries": 1, "Sandwiches": 2,
}
DEFAULT_SHELF_PERISHABLE = 4
DEFAULT_SHELF_NONPERISHABLE = 180

PERISHABLE_SUBCATS = set(SHELF_LIFE.keys())

# Pincode -> (lat, lon) via api.zippopotam.us/IN (verified 30-09-2026).
# Cached here to avoid re-hitting geocode API on every run.
PINCODE_GEO = {
    110001: (28.6333, 77.2167),   # Delhi - Janpath (Zippopotam verified)
    160001: (30.7372, 76.7872),   # Chandigarh - New Secretariate (Zippopotam)
    226001: (27.5167, 79.9833),   # Lalbagh, UP (Zippopotam; was Lucknow approx)
    302001: (26.1748, 74.8799),   # Jaipur 1 GPO (Zippopotam; was Jaipur approx)
    # 380001 Zippopotam returns 30.25,77.58 (wrong, points to Haryana) -
    # keep Ahmedabad correct coords:
    380001: (23.0333, 72.6167),   # Ahmedabad, Gujarat
    400001: (19.0167, 72.85),     # Mumbai (Zippopotam verified)
    # 462001 Zippopotam 404 - keep Bhopal:
    462001: (23.2667, 77.40),     # Bhopal, MP
    560001: (12.2667, 76.6833),   # Karnataka Rajbhavan (Zippopotam)
    600001: (13.0656, 80.2672),   # Chennai Flower Bazar (Zippopotam)
    700001: (22.5690, 88.3697),   # Kolkata Lalbazar (Zippopotam)
}

# WMO weathercode -> label (Open-Meteo)
WMO_MAP = {
    0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog", 51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light showers", 81: "Showers", 82: "Heavy showers",
    95: "Thunderstorm", 96: "Storm+hail", 99: "Storm+heavy hail",
}
