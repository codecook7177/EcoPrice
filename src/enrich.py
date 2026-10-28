"""EcoPrice enrichment: geocode (Zippopotam) + weather (Open-Meteo) + festival (holidays-IN).

Why these APIs (best free, no key, verified 30-09-2026):
- Geocode: https://api.zippopotam.us/IN/<pincode> -> lat/lon. Free, no key.
- Weather history: https://archive-api.open-meteo.com/v1/archive (ERA5).
  1 call per pincode covers 2019-01-01..2023-12-31 daily means. Free, no key,
  timezone=Asia/Kolkata. Used at SALE date (not vague dataset cols).
- Festival/holiday: `holidays` lib, country=IN (offline, deterministic).
  Covers Diwali, Holi, Eid, Dussehra, Christmas etc. Nager.Date returns 204
  for IN (unsupported), Calendarific/WeatherAPI need paid keys, so holidays
  lib is the reproducible choice. Exposes is_holiday + festival_name.

Dataset reality check:
- Kaggle abuhumzakhan/store-data has NO weather/event/time columns
  (cols: Sales Date, Order Date, Postal Code, ...). User memory of vague
  cols is stale, so nothing to drop; we IGNORЕ any such col if present and
  re-derive everything from Postal Code + Sales Date via APIs above.
- No Time column -> hourly weather fixed at 12:00 IST proxy; pipeline uses
  DAILY means (temperature_2m_mean etc.) so time-of-day is irrelevant.
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd

from .config import DATA, PINCODE_GEO

WEATHER_CACHE = DATA / "weather_daily.csv"
GEO_CACHE = DATA / "pincode_geo.csv"


def fetch_latlon(pincode: int, timeout=15):
    if int(pincode) in PINCODE_GEO:
        return PINCODE_GEO[int(pincode)]
    url = f"https://api.zippopotam.us/IN/{int(pincode)}"
    req = urllib.request.Request(url, headers={"User-Agent": "EcoPrice/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.loads(r.read().decode())
        p = j["places"][0]
        return float(p["latitude"]), float(p["longitude"])


def fetch_weather_daily(lat, lon, start="2019-01-01", end="2023-12-31", timeout=60):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
        "&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,weathercode"
        "&timezone=Asia%2FKolkata"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "EcoPrice/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.loads(r.read().decode())
    d = j["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "temperature": d["temperature_2m_mean"],
        "humidity": d["relative_humidity_2m_mean"],
        "precipitation": d["precipitation_sum"],
        "weathercode": d["weathercode"],
    })


def build_weather_cache(pincodes, force=False):
    """One Open-Meteo call per pincode. Cached to data/weather_daily.csv."""
    if WEATHER_CACHE.exists() and not force:
        return pd.read_csv(WEATHER_CACHE, parse_dates=["date"])
    frames = []
    geo_rows = []
    for pin in sorted(set(int(p) for p in pincodes)):
        lat, lon = fetch_latlon(pin)
        geo_rows.append({"postal_code": pin, "lat": lat, "lon": lon})
        w = fetch_weather_daily(lat, lon)
        w["postal_code"] = pin
        frames.append(w)
        print(f"weather: {pin} ({lat},{lon}) -> {len(w)} days")
    out = pd.concat(frames, ignore_index=True)
    DATA.mkdir(parents=True, exist_ok=True)
    out.to_csv(WEATHER_CACHE, index=False)
    pd.DataFrame(geo_rows).to_csv(GEO_CACHE, index=False)
    return out


def add_holidays(df, date_col="sales_date", window=7, far_value=8):
    """Add is_holiday + festival_name + days_to_festival countdown.

    days_to_festival: 0 = festival today, 1..7 = festival in N days,
    8 = no festival within `window` days (don't consider).
    Looks ahead to next year so Dec dates still see Jan festivals.
    """
    import bisect
    import holidays
    dates = pd.to_datetime(df[date_col])
    yrs = sorted(dates.dt.year.unique().tolist())
    # +1 year lookahead for year-end countdown (e.g. 2023-12-30 -> 2024-01-01)
    cal = holidays.India(years=yrs + [max(yrs) + 1])
    hdays = sorted(cal.keys())  # list[date]
    hnames = {d: cal.get(d, "") or "" for d in hdays}

    s = dates.dt.date
    fest, ishol, countdown, upcoming = [], [], [], []
    for d in s:
        name = cal.get(d, "") or ""
        fest.append(name)
        ishol.append(1 if name else 0)
        i = bisect.bisect_left(hdays, d)
        if i < len(hdays):
            delta = (hdays[i] - d).days
            if delta <= window:
                countdown.append(int(delta))
                upcoming.append(hnames[hdays[i]])
            else:
                countdown.append(far_value)
                upcoming.append("")
        else:
            countdown.append(far_value)
            upcoming.append("")
    df["festival_name"] = fest
    df["days_to_festival"] = countdown
    df["upcoming_festival"] = upcoming
    # Invariant: is_holiday == (days_to_festival == 0)
    df["is_holiday"] = (df["days_to_festival"] == 0).astype(int)
    return df
