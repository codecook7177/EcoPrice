"""
EcoPrice - Phase 1: Data Preprocessing
Kerala Kirana Store POS Data -> Clean, enriched dataset ready for feature extraction.

Steps:
  1. Load raw POS data
  2. Clean columns, parse dates, handle bad values
  3. Normalize product names & categorize into perishable groups
  4. Pull historical weather data from Open-Meteo (Kerala, Jan-Mar 2025)
  5. Add festival/holiday flags using holidays-IN library
  6. Aggregate to daily level per product category
  7. Save cleaned outputs
"""

import pandas as pd
import numpy as np
import json
import os
import time

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_FILE = os.path.join(DATA_DIR, "sales_register_data.csv")
CLEAN_FILE = os.path.join(DATA_DIR, "kerala_clean.csv")
DAILY_FILE = os.path.join(DATA_DIR, "kerala_daily.csv")
REPORT_FILE = os.path.join(DATA_DIR, "preprocess_report_v2.json")

# ---------------------------------------------------------------------------
# STATE SUBDIVISIONS & LOCATION RESOLUTION FOR WEATHER AND HOLIDAYS
# ---------------------------------------------------------------------------
STATE_SUBDIV_MAP = {
    "Andhra Pradesh": "AP",
    "Karnataka": "KA",
    "Kerala": "KL",
    "Tamil Nadu": "TN",
    "Telangana": "TS",
    "Maharashtra": "MH",
    "Delhi": "DL",
    "Gujarat": "GJ",
    "West Bengal": "WB",
    "Uttar Pradesh": "UP",
    "Rajasthan": "RJ",
    "Goa": "GA",
    "Punjab": "PB",
    "Haryana": "HR",
    "Madhya Pradesh": "MP",
    "Bihar": "BR",
    "Assam": "AS",
    "Odisha": "OD",
}

def resolve_location_metadata(city_name: str = None):
    """
    Resolves the store city, state, subdivision code for Indian holidays,
    and geographical coordinates for Open-Meteo historical weather archives.
    """
    from src.weather_service import CITY_PROFILES

    if not city_name:
        try:
            settings_path = os.path.join(DATA_DIR, "settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, "r", encoding="utf-8") as f:
                    s = json.load(f)
                    city_name = s.get("store_location") or s.get("location")
        except Exception:
            pass

    if not city_name:
        city_name = "Visakhapatnam"

    clean = city_name.strip().split(",")[0].strip().title()
    profile = CITY_PROFILES.get(clean)
    if not profile:
        matched = next((k for k in CITY_PROFILES if k.lower() in clean.lower()), None)
        profile = CITY_PROFILES.get(matched) if matched else None

    if profile:
        city = profile["city"]
        state = profile.get("state", "Andhra Pradesh")
        lat = profile.get("lat", 17.6868)
        lon = profile.get("lon", 83.2185)
    else:
        city = clean
        state = "Andhra Pradesh"
        lat = 17.6868
        lon = 83.2185

    subdiv = STATE_SUBDIV_MAP.get(state, "AP")
    return city, state, subdiv, lat, lon

# ---------------------------------------------------------------------------
# PERISHABLE CATEGORY MAPPING
# ---------------------------------------------------------------------------
# Map product name keywords -> category
CATEGORY_MAP = {
    # Dairy
    "MILK": "Dairy", "CURD": "Dairy", "BUTTER": "Dairy", "GHEE": "Dairy",
    "CHEESE": "Dairy", "PANEER": "Dairy", "CREAM": "Dairy", "YOGURT": "Dairy",
    "DAHI": "Dairy", "LASSI": "Dairy", "MILMA": "Dairy",
    # Eggs
    "EGG": "Eggs",
    # Fruits
    "APPLE": "Fruits", "BANANA": "Fruits", "ORANGE": "Fruits", "MANGO": "Fruits",
    "GRAPE": "Fruits", "PAPAYA": "Fruits", "PINEAPPLE": "Fruits",
    "WATERMELON": "Fruits", "LEMON": "Fruits", "LIME": "Fruits",
    "POMEGRANATE": "Fruits", "GUAVA": "Fruits", "CHIKOO": "Fruits",
    "MOSAMBI": "Fruits", "KIWI": "Fruits", "PEAR": "Fruits",
    "JACKFRUIT": "Fruits", "COCONUT": "Fruits",
    # Vegetables
    "ONION": "Vegetables", "TOMATO": "Vegetables", "POTATO": "Vegetables",
    "CARROT": "Vegetables", "BEANS": "Vegetables", "CABBAGE": "Vegetables",
    "CAULIFLOWER": "Vegetables", "CUCUMBER": "Vegetables", "GINGER": "Vegetables",
    "GARLIC": "Vegetables", "CHILLY": "Vegetables", "CHILLI": "Vegetables",
    "BRINJAL": "Vegetables", "CAPSICUM": "Vegetables", "BEETROOT": "Vegetables",
    "RADISH": "Vegetables", "SPINACH": "Vegetables", "CORIANDER": "Vegetables",
    "CURRY LEAF": "Vegetables", "DRUMSTICK": "Vegetables", "LADIES": "Vegetables",
    "OKRA": "Vegetables", "PEAS": "Vegetables", "CORN": "Vegetables",
    "MUSHROOM": "Vegetables", "GOURD": "Vegetables", "BITTER": "Vegetables",
    "SNAKE": "Vegetables", "RIDGE": "Vegetables", "ULLI": "Vegetables",
    "SWEE POTATO": "Vegetables", "RAW BANANA": "Vegetables",
    "MALLILLA": "Vegetables", "KIRAN MATHA": "Vegetables",
    "GREEN CHILLY": "Vegetables", "BIG ONION": "Vegetables",
    "WHITE GARLIC": "Vegetables",
    # Bakery
    "BREAD": "Bakery", "BUN": "Bakery", "CAKE": "Bakery", "BISCUIT": "Bakery",
    "COOKIE": "Bakery", "RUSK": "Bakery", "TOAST": "Bakery",
    "PAPPDAM": "Bakery", "PATHIRI": "Bakery",
    # Meat & Fish
    "CHICKEN": "Meat & Fish", "FISH": "Meat & Fish", "MUTTON": "Meat & Fish",
    "BEEF": "Meat & Fish", "PRAWN": "Meat & Fish",
    # Pulses & Grains (semi-perishable)
    "RICE": "Pulses & Grains", "DAL": "Pulses & Grains", "ATTA": "Pulses & Grains",
    "WHEAT": "Pulses & Grains", "MAIDA": "Pulses & Grains", "RAVA": "Pulses & Grains",
    "PAYAR": "Pulses & Grains", "KADALA": "Pulses & Grains",
    "CHANA": "Pulses & Grains", "MOONG": "Pulses & Grains",
    "TOOR": "Pulses & Grains", "URAD": "Pulses & Grains",
    # Snacks & Packaged (less perishable but included)
    "SNACKS": "Snacks", "CHIPS": "Snacks", "MIXTURE": "Snacks",
    "NAMKEEN": "Snacks", "POKA": "Snacks", "NEDRA": "Snacks",
    "ROBEST": "Snacks",
}


def classify_product(product_name):
    """Classify a product into a perishable category."""
    name_upper = str(product_name).upper().strip()
    # Try exact multi-word matches first
    for keyword, category in CATEGORY_MAP.items():
        if keyword in name_upper:
            return category
    return "Other"


def fetch_weather_data(lat, lon, start_date, end_date):
    """Fetch historical daily weather from Open-Meteo."""
    import urllib.request
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        f"precipitation_sum,rain_sum,relative_humidity_2m_mean,"
        f"wind_speed_10m_max"
        f"&timezone=Asia/Kolkata"
    )
    print(f"  Fetching weather: {start_date} to {end_date}...")
    req = urllib.request.Request(url, headers={"User-Agent": "EcoPrice/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    weather_df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "temp_max": data["daily"]["temperature_2m_max"],
        "temp_min": data["daily"]["temperature_2m_min"],
        "temp_mean": data["daily"]["temperature_2m_mean"],
        "precipitation": data["daily"]["precipitation_sum"],
        "rain": data["daily"]["rain_sum"],
        "humidity": data["daily"]["relative_humidity_2m_mean"],
        "wind_speed": data["daily"]["wind_speed_10m_max"],
    })
    print(f"  Got {len(weather_df)} days of weather data")
    return weather_df


def get_holidays(start_date, end_date, subdiv="AP", state_name="Andhra Pradesh"):
    """Get Indian national and state-specific festival holidays across all dataset years."""
    import holidays as hlib
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    years = list(range(start_dt.year, end_dt.year + 1))
    if not years:
        years = [2025]

    try:
        india = hlib.India(years=years, subdiv=subdiv)
    except Exception:
        india = hlib.India(years=years)

    dates = pd.date_range(start_date, end_date)
    records = []
    for d in dates:
        is_hol = d in india
        hol_name = india.get(d, "")
        records.append({
            "date": d,
            "is_holiday": int(is_hol),
            "holiday_name": hol_name,
        })
    hol_df = pd.DataFrame(records)
    n_holidays = hol_df["is_holiday"].sum()
    print(f"  Found {n_holidays} holidays in range for {state_name} ({subdiv}) across years {years}")
    return hol_df


def main(city_name=None, raw_path=None):
    city, state, subdiv, lat, lon = resolve_location_metadata(city_name)
    print("=" * 60)
    print("EcoPrice Phase 1: PREPROCESSING & ENRICHMENT")
    print(f"Store Location: {city}, {state} (Coordinates: {lat}N, {lon}E | Holiday Subdiv: {subdiv})")
    print("=" * 60)

    # ------------------------------------------------------------------
    # STEP 1: Load raw data
    # ------------------------------------------------------------------
    print("\n[STEP 1] Loading raw data...")
    source_file = raw_path or RAW_FILE
    df = pd.read_csv(source_file, encoding="latin1")
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    report = {
        "raw_rows": len(df),
        "raw_columns": len(df.columns),
    }

    # ------------------------------------------------------------------
    # STEP 2: Clean columns & parse dates
    # ------------------------------------------------------------------
    print("\n[STEP 2] Cleaning data...")

    # Identify date column flexibly
    date_col = next((c for c in df.columns if str(c).strip().lower() in ["date", "datetime", "bill_date", "invoice_date", "timestamp"]), "Date")

    # Parse dates robustly (handles day/month/year, year-month-day, ISO datetimes, mixed formats)
    try:
        df["parsed_dt"] = pd.to_datetime(df[date_col], format="%d/%m/%Y", dayfirst=True)
    except Exception:
        df["parsed_dt"] = pd.to_datetime(df[date_col], format="mixed", dayfirst=True, errors="coerce")

    # Date component (day level)
    df["date"] = df["parsed_dt"].dt.floor("D")

    # Extract transaction hour if explicit time column or timestamp exists
    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    time_col = next((cols_lower[k] for k in ["time", "timestamp", "bill_time", "billtime", "txn_time", "transaction_time", "created_at", "clock_time", "hour"] if k in cols_lower), None)

    if time_col and time_col in df.columns:
        try:
            num_h = pd.to_numeric(df[time_col], errors="coerce")
            if num_h.notna().mean() > 0.5 and (num_h.between(0, 23).mean() > 0.8):
                df["hour"] = num_h.fillna(12).astype(int)
            else:
                parsed_t = pd.to_datetime(df[time_col].astype(str), format="mixed", errors="coerce")
                df["hour"] = parsed_t.dt.hour.fillna(12).astype(int)
            print(f"  Extracted transaction hour from column '{time_col}'")
        except Exception as e:
            print(f"  Note on parsing time column '{time_col}': {e}")
            df["hour"] = None
    elif df["parsed_dt"].dt.hour.nunique() > 1 or (df["parsed_dt"].dt.hour > 0).any():
        df["hour"] = df["parsed_dt"].dt.hour.fillna(12).astype(int)
        print("  Extracted transaction hour from combined datetime column")
    else:
        df["hour"] = None

    # Clean numeric columns that have commas
    for col in ["Net Amount", "Taxable Amount"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ""), errors="coerce"
        )

    # Clean Qty - remove zero or negative
    before = len(df)
    df = df[df["Qty"] > 0].copy()
    removed_qty = before - len(df)
    print(f"  Removed {removed_qty} rows with Qty <= 0")

    # Clean Rate - remove zero or negative
    before = len(df)
    df = df[df["Rate"] > 0].copy()
    removed_rate = before - len(df)
    print(f"  Removed {removed_rate} rows with Rate <= 0")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    removed_dup = before - len(df)
    print(f"  Removed {removed_dup} duplicate rows")

    # Rename columns for consistency
    df = df.rename(columns={
        "Product Name": "product_name",
        "Party Name": "party_name",
        "Pur No": "invoice_no",
    })

    print(f"  Remaining: {len(df):,} rows")
    report["cleaned_rows"] = len(df)
    report["removed_zero_qty"] = removed_qty
    report["removed_zero_rate"] = removed_rate
    report["removed_duplicates"] = removed_dup

    # ------------------------------------------------------------------
    # STEP 3: Normalize product names & categorize
    # ------------------------------------------------------------------
    print("\n[STEP 3] Categorizing products...")

    # Strip and upper-case product names
    df["product_name"] = df["product_name"].str.strip().str.upper()

    # Classify into perishable categories
    df["category"] = df["product_name"].apply(classify_product)

    cat_counts = df["category"].value_counts()
    print("  Category distribution:")
    for cat, count in cat_counts.items():
        pct = count / len(df) * 100
        print(f"    {cat:20s}: {count:>6,} ({pct:5.1f}%)")

    # Keep only perishable categories (drop "Other")
    perishable_cats = [c for c in cat_counts.index if c != "Other"]
    perishable_mask = df["category"].isin(perishable_cats)
    df_perish = df[perishable_mask].copy()
    print(f"\n  Perishable rows: {len(df_perish):,} ({len(df_perish)/len(df)*100:.1f}%)")
    print(f"  Non-perishable (dropped): {len(df) - len(df_perish):,}")

    report["perishable_rows"] = len(df_perish)
    report["categories"] = cat_counts.to_dict()

    # ------------------------------------------------------------------
    # STEP 4: Fetch weather data
    # ------------------------------------------------------------------
    print(f"\n[STEP 4] Fetching weather data (Open-Meteo, {city}, {state})...")
    start_date = df_perish["date"].min().strftime("%Y-%m-%d")
    end_date = df_perish["date"].max().strftime("%Y-%m-%d")

    try:
        weather_df = fetch_weather_data(lat, lon, start_date, end_date)
        weather_df.to_csv(os.path.join(DATA_DIR, "kerala_weather.csv"), index=False)
        print("  Saved to data/kerala_weather.csv")
        report["weather_source"] = "Open-Meteo Archive API"
        report["weather_location"] = f"{city}, {state} ({lat}N, {lon}E)"
        report["weather_days"] = len(weather_df)
    except Exception as e:
        print(f"  WARNING: Weather fetch failed: {e}")
        # Do not substitute constant "profile" values: they look like real
        # weather but carry no information. Leave the columns empty so that
        # feature extraction drops them.
        print("  Weather columns left empty; they will be dropped from the features.")
        dates_range = pd.date_range(start_date, end_date)
        weather_df = pd.DataFrame({"date": dates_range})
        for col in ["temp_max", "temp_min", "temp_mean", "precipitation", "rain", "humidity", "wind_speed"]:
            weather_df[col] = np.nan
        report["weather_source"] = "unavailable (fetch failed)"

    # ------------------------------------------------------------------
    # STEP 5: Add holidays/festivals
    # ------------------------------------------------------------------
    print(f"\n[STEP 5] Adding {state} ({subdiv}) and Indian National festival flags...")
    try:
        hol_df = get_holidays(start_date, end_date, subdiv=subdiv, state_name=state)
        report["holidays_found"] = int(hol_df["is_holiday"].sum())
        holiday_names = [name for name in hol_df[hol_df["is_holiday"] == 1]["holiday_name"].tolist() if name]
        report["holiday_names"] = holiday_names
        print("  Holidays identified: " + (", ".join(holiday_names[:8]) or "None in date window"))
    except Exception as e:
        print(f"  WARNING: Holiday fetch failed: {e}")
        dates_range = pd.date_range(start_date, end_date)
        hol_df = pd.DataFrame({
            "date": dates_range,
            "is_holiday": 0,
            "holiday_name": "",
        })

    # ------------------------------------------------------------------
    # STEP 6: Aggregate to daily level per category
    # ------------------------------------------------------------------
    print("\n[STEP 6] Aggregating to daily level per category...")

    daily = df_perish.groupby(["date", "category"]).agg(
        total_qty=("Qty", "sum"),
        total_revenue=("Net Amount", "sum"),
        avg_price=("Rate", "mean"),
        n_transactions=("Qty", "count"),
        n_unique_products=("product_name", "nunique"),
        avg_qty_per_txn=("Qty", "mean"),
    ).reset_index()

    # Add time features
    daily["day_of_week"] = daily["date"].dt.dayofweek  # 0=Mon, 6=Sun
    daily["day_name"] = daily["date"].dt.day_name()
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    daily["day_of_month"] = daily["date"].dt.day
    daily["month"] = daily["date"].dt.month
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["is_month_start"] = (daily["day_of_month"] <= 5).astype(int)
    daily["is_month_end"] = (daily["day_of_month"] >= 25).astype(int)

    # Merge weather
    daily = daily.merge(weather_df, on="date", how="left")

    # Merge holidays
    daily = daily.merge(hol_df[["date", "is_holiday", "holiday_name"]], on="date", how="left")
    daily["is_holiday"] = daily["is_holiday"].fillna(0).astype(int)

    print(f"  Daily aggregated rows: {len(daily):,}")
    print(f"  Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"  Categories: {daily['category'].nunique()}")

    report["daily_rows"] = len(daily)
    report["date_range"] = f"{daily['date'].min().date()} to {daily['date'].max().date()}"
    report["daily_columns"] = list(daily.columns)

    # ------------------------------------------------------------------
    # STEP 7: Save outputs
    # ------------------------------------------------------------------
    print("\n[STEP 7] Saving outputs...")

    # Save full clean transaction-level data
    clean_cols = ["date", "product_name", "category", "Qty", "Rate", "GST",
                  "Net Amount", "Taxable Amount", "party_name", "invoice_no"]
    if "hour" in df_perish.columns and df_perish["hour"].notna().any():
        clean_cols.append("hour")
    df_perish[clean_cols].to_csv(CLEAN_FILE, index=False)
    print(f"  Saved {CLEAN_FILE} ({len(df_perish):,} rows)")

    # Save daily aggregated data
    daily.to_csv(DAILY_FILE, index=False)
    print(f"  Saved {DAILY_FILE} ({len(daily):,} rows)")

    # Save report
    # Convert numpy types for JSON
    for k, v in report.items():
        if isinstance(v, (np.integer,)):
            report[k] = int(v)
        elif isinstance(v, dict):
            report[k] = {str(kk): int(vv) if isinstance(vv, (np.integer,)) else vv
                         for kk, vv in v.items()}

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved {REPORT_FILE}")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"  Raw input:          {report['raw_rows']:>8,} rows")
    print(f"  After cleaning:     {report['cleaned_rows']:>8,} rows")
    print(f"  Perishable only:    {report['perishable_rows']:>8,} rows")
    print(f"  Daily aggregated:   {report['daily_rows']:>8,} rows")
    print(f"  Date range:         {report['date_range']}")
    print(f"  Weather:            {report.get('weather_days', 'N/A')} days from Open-Meteo")
    print(f"  Holidays:           {report.get('holidays_found', 'N/A')} found")
    print()
    print("  Output files:")
    print(f"    data/kerala_clean.csv   (transaction-level)")
    print(f"    data/kerala_daily.csv   (daily aggregated + weather + holidays)")
    print(f"    data/kerala_weather.csv (raw weather cache)")
    print()

    # Quick sanity: show daily qty distribution
    print("  Daily total_qty distribution (our TARGET):")
    tq = daily["total_qty"]
    print(f"    Min: {tq.min():.1f}, Max: {tq.max():.1f}")
    print(f"    Mean: {tq.mean():.1f}, Median: {tq.median():.1f}, Std: {tq.std():.1f}")
    print(f"    Skewness: {tq.skew():.3f}")
    print()

    # Show Sunday vs Monday difference
    sun = daily[daily["day_of_week"] == 6]["total_qty"].mean()
    mon = daily[daily["day_of_week"] == 0]["total_qty"].mean()
    print(f"  Sunday avg qty:  {sun:.1f}")
    print(f"  Monday avg qty:  {mon:.1f}")
    print(f"  Monday/Sunday ratio: {mon/sun:.1f}x  (strong weekday pattern!)")


if __name__ == "__main__":
    main()
