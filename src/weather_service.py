"""
EcoPrice - Weather & Climate Engine
Provides real-time and high-fidelity meteorological intelligence for store locations across India.
Calculates weather-driven perishable spoilage acceleration factors (k_spoil)
and feeds environmental observations (temp, humidity, precip) to the PPO Dynamic Pricing Agent.
Includes GPS and IP location auto-detection.
"""

import math
import time
from typing import Dict, Any, List

CITY_PROFILES: Dict[str, Dict[str, Any]] = {
    "Kochi": {
        "city": "Kochi",
        "state": "Kerala",
        "lat": 9.9312,
        "lon": 76.2673,
        "base_temp": 31.0,
        "base_humidity": 82.0,
        "base_precip": 1.2,
        "climate": "Coastal Tropical Humid",
    },
    "Thiruvananthapuram": {
        "city": "Thiruvananthapuram",
        "state": "Kerala",
        "lat": 8.5241,
        "lon": 76.9366,
        "base_temp": 30.5,
        "base_humidity": 80.0,
        "base_precip": 0.8,
        "climate": "Southern Coastal Humid",
    },
    "Kozhikode": {
        "city": "Kozhikode",
        "state": "Kerala",
        "lat": 11.2588,
        "lon": 75.7804,
        "base_temp": 30.2,
        "base_humidity": 83.0,
        "base_precip": 1.0,
        "climate": "Malabar Maritime",
    },
    "Bengaluru": {
        "city": "Bengaluru",
        "state": "Karnataka",
        "lat": 12.9716,
        "lon": 77.5946,
        "base_temp": 24.5,
        "base_humidity": 55.0,
        "base_precip": 0.0,
        "climate": "Moderate Plateau",
    },
    "Mysuru": {
        "city": "Mysuru",
        "state": "Karnataka",
        "lat": 12.2958,
        "lon": 76.6394,
        "base_temp": 26.5,
        "base_humidity": 58.0,
        "base_precip": 0.0,
        "climate": "South Deccan Plateau",
    },
    "Chennai": {
        "city": "Chennai",
        "state": "Tamil Nadu",
        "lat": 13.0827,
        "lon": 80.2707,
        "base_temp": 34.2,
        "base_humidity": 78.0,
        "base_precip": 0.0,
        "climate": "Tropical Coastal Heat",
    },
    "Coimbatore": {
        "city": "Coimbatore",
        "state": "Tamil Nadu",
        "lat": 11.0168,
        "lon": 76.9558,
        "base_temp": 30.5,
        "base_humidity": 64.0,
        "base_precip": 0.0,
        "climate": "Western Ghats Inland",
    },
    "Madurai": {
        "city": "Madurai",
        "state": "Tamil Nadu",
        "lat": 9.9252,
        "lon": 78.1198,
        "base_temp": 35.0,
        "base_humidity": 62.0,
        "base_precip": 0.0,
        "climate": "Southern Inland Plain",
    },
    "Mumbai": {
        "city": "Mumbai",
        "state": "Maharashtra",
        "lat": 19.0760,
        "lon": 72.8777,
        "base_temp": 32.0,
        "base_humidity": 77.0,
        "base_precip": 0.0,
        "climate": "Humid Maritime",
    },
    "Pune": {
        "city": "Pune",
        "state": "Maharashtra",
        "lat": 18.5204,
        "lon": 73.8567,
        "base_temp": 27.8,
        "base_humidity": 56.0,
        "base_precip": 0.0,
        "climate": "Inland Semi-Arid",
    },
    "Delhi": {
        "city": "Delhi",
        "state": "NCR",
        "lat": 28.6139,
        "lon": 77.2090,
        "base_temp": 33.8,
        "base_humidity": 45.0,
        "base_precip": 0.0,
        "climate": "Semi-Arid Continental",
    },
    "Hyderabad": {
        "city": "Hyderabad",
        "state": "Telangana",
        "lat": 17.3850,
        "lon": 78.4867,
        "base_temp": 30.0,
        "base_humidity": 60.0,
        "base_precip": 0.0,
        "climate": "Deccan Plateau Warm",
    },
    "Kolkata": {
        "city": "Kolkata",
        "state": "West Bengal",
        "lat": 22.5726,
        "lon": 88.3639,
        "base_temp": 32.5,
        "base_humidity": 81.0,
        "base_precip": 0.4,
        "climate": "Delta Tropical Humid",
    },
    "Ahmedabad": {
        "city": "Ahmedabad",
        "state": "Gujarat",
        "lat": 23.0225,
        "lon": 72.5714,
        "base_temp": 35.5,
        "base_humidity": 42.0,
        "base_precip": 0.0,
        "climate": "Dry Heat",
    },
    "Surat": {
        "city": "Surat",
        "state": "Gujarat",
        "lat": 21.1702,
        "lon": 72.8311,
        "base_temp": 33.0,
        "base_humidity": 68.0,
        "base_precip": 0.0,
        "climate": "Western Coastal Plain",
    },
    "Jaipur": {
        "city": "Jaipur",
        "state": "Rajasthan",
        "lat": 26.9124,
        "lon": 75.7873,
        "base_temp": 34.5,
        "base_humidity": 38.0,
        "base_precip": 0.0,
        "climate": "Arid Continental",
    },
    "Goa": {
        "city": "Goa",
        "state": "Goa",
        "lat": 15.2993,
        "lon": 74.1240,
        "base_temp": 31.5,
        "base_humidity": 80.0,
        "base_precip": 0.5,
        "climate": "Konkan Coastal",
    },
    "Visakhapatnam": {
        "city": "Visakhapatnam",
        "state": "Andhra Pradesh",
        "lat": 17.6868,
        "lon": 83.2185,
        "base_temp": 27.0,
        "base_humidity": 82.0,
        "base_precip": 1.2,
        "climate": "Eastern Coastal Bay",
    },
}


class WeatherService:
    _instance = None
    _cache: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WeatherService, cls).__new__(cls)
        return cls._instance

    def detect_nearest_city(self, lat: float, lon: float) -> str:
        """Finds closest supported retail city using Great-Circle distance."""
        best_city = "Kochi"
        min_dist = float("inf")

        for name, p in CITY_PROFILES.items():
            d_lat = math.radians(p["lat"] - lat)
            d_lon = math.radians(p["lon"] - lon)
            a = math.sin(d_lat / 2)**2 + math.cos(math.radians(lat)) * math.cos(math.radians(p["lat"])) * math.sin(d_lon / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            dist = 6371 * c
            if dist < min_dist:
                min_dist = dist
                best_city = name

        return best_city

    def reverse_geocode(self, lat: float, lon: float) -> tuple:
        """Attempts server-side reverse geocoding to resolve exact town/locality."""
        import urllib.request
        import json

        # Attempt 1: BigDataCloud (fast, client-focused, free)
        try:
            url = f"https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={lat}&longitude={lon}&localityLanguage=en"
            req = urllib.request.Request(url, headers={"User-Agent": "EcoPrice/1.0"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode())
                locality = data.get("locality") or data.get("city")
                state = data.get("principalSubdivision", "")
                if locality:
                    return locality.strip(), state.strip()
        except Exception:
            pass

        # Attempt 2: OpenStreetMap Nominatim
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=14"
            req = urllib.request.Request(url, headers={"User-Agent": "EcoPriceApp/1.0"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode())
                addr = data.get("address", {})
                place = addr.get("village") or addr.get("town") or addr.get("suburb") or addr.get("city") or addr.get("county")
                state = addr.get("state", "")
                if place:
                    return place.strip(), state.strip()
        except Exception:
            pass

        return None, None

    def register_detected_gps_location(
        self,
        lat: float,
        lon: float,
        suggested_city: str = "",
        suggested_state: str = ""
    ) -> tuple:
        """
        Registers an exact GPS-detected store location.
        Uses exact locality name (e.g. Ibrahimpatnam) instead of forcing nearest hub name.
        """
        nearest_hub = self.detect_nearest_city(lat, lon)
        hub_profile = CITY_PROFILES.get(nearest_hub, CITY_PROFILES["Kochi"])

        city_name = (suggested_city or "").strip()
        state_name = (suggested_state or "").strip()

        if not city_name:
            geo_city, geo_state = self.reverse_geocode(lat, lon)
            if geo_city:
                city_name = geo_city
                state_name = geo_state or state_name

        if not city_name:
            city_name = nearest_hub
            state_name = hub_profile["state"]
        elif not state_name:
            state_name = hub_profile["state"]

        # Fetch live telemetry for coordinates from Open-Meteo
        live_temp = hub_profile["base_temp"]
        live_humid = hub_profile["base_humidity"]
        live_precip = hub_profile["base_precip"]

        try:
            import urllib.request
            import json
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation"
            req = urllib.request.Request(url, headers={"User-Agent": "EcoPrice/1.0"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode())
                curr = data.get("current", {})
                if "temperature_2m" in curr and curr["temperature_2m"] is not None:
                    live_temp = float(curr["temperature_2m"])
                if "relative_humidity_2m" in curr and curr["relative_humidity_2m"] is not None:
                    live_humid = float(curr["relative_humidity_2m"])
                if "precipitation" in curr and curr["precipitation"] is not None:
                    live_precip = float(curr["precipitation"])
        except Exception:
            pass

        # Climate description combines exact locality and regional climate classification
        climate_tag = f"{hub_profile['climate']} ({nearest_hub} Met Zone)"

        profile = {
            "city": city_name,
            "state": state_name,
            "lat": lat,
            "lon": lon,
            "base_temp": round(live_temp, 1),
            "base_humidity": round(live_humid, 1),
            "base_precip": round(live_precip, 2),
            "climate": climate_tag,
        }

        # Store in CITY_PROFILES so it remains selected and accessible everywhere
        CITY_PROFILES[city_name] = profile

        weather = self.get_weather(city_name)
        return city_name, state_name, weather

    def _fetch_live_telemetry(self, lat: float, lon: float, default_temp: float, default_humid: float, default_precip: float) -> tuple:
        """
        Fetches live real-time ambient weather from Open-Meteo for exact latitude/longitude.
        Caches responses for 10 minutes (600s) to keep page loads instantaneous (0ms).
        Falls back to regional seasonal profiles if network is temporarily unreachable.
        """
        now = time.time()
        cache_key = f"{round(lat, 2)}_{round(lon, 2)}"
        cached = self._cache.get(cache_key)
        if cached and (now - cached.get("timestamp", 0) < 600):
            return cached["temp"], cached["humidity"], cached["precip"]

        live_temp = default_temp
        live_humid = default_humid
        live_precip = default_precip

        try:
            import urllib.request
            import json
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation"
            req = urllib.request.Request(url, headers={"User-Agent": "EcoPrice/1.0"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode())
                curr = data.get("current", {})
                if "temperature_2m" in curr and curr["temperature_2m"] is not None:
                    live_temp = float(curr["temperature_2m"])
                if "relative_humidity_2m" in curr and curr["relative_humidity_2m"] is not None:
                    live_humid = float(curr["relative_humidity_2m"])
                if "precipitation" in curr and curr["precipitation"] is not None:
                    live_precip = float(curr["precipitation"])

                self._cache[cache_key] = {
                    "temp": round(live_temp, 1),
                    "humidity": round(live_humid, 1),
                    "precip": round(live_precip, 2),
                    "timestamp": now,
                }
        except Exception:
            pass

        return round(live_temp, 1), round(live_humid, 1), round(live_precip, 2)

    def get_weather(self, city_name: str = "Kochi") -> Dict[str, Any]:
        """
        Retrieves meteorological telemetry for the selected store city.
        Queries live Open-Meteo real-time telemetry (cached for 10 mins).
        Calculates Arrhenius biological decay multiplier.
        """
        clean_city = (city_name or "Kochi").strip().split(",")[0].strip().title()

        profile = CITY_PROFILES.get(clean_city)
        if not profile:
            matched = next((k for k in CITY_PROFILES if k.lower() in clean_city.lower()), None)
            if matched:
                profile = CITY_PROFILES[matched]
                clean_city = profile["city"]
            else:
                # Dynamically construct profile for custom user-entered city
                profile = {
                    "city": clean_city,
                    "state": "Retail Store Location",
                    "lat": 15.0,
                    "lon": 77.0,
                    "base_temp": 28.0,
                    "base_humidity": 70.0,
                    "base_precip": 0.0,
                    "climate": "Custom Retail Zone",
                }

        # Query live real-time weather from Open-Meteo (with 10-min cache and local fallback)
        temp, humidity, precip = self._fetch_live_telemetry(
            profile.get("lat", 17.0),
            profile.get("lon", 78.0),
            profile.get("base_temp", 28.0),
            profile.get("base_humidity", 70.0),
            profile.get("base_precip", 0.0)
        )
        climate = profile["climate"]

        # Calculate Spoilage Acceleration Factor (Arrhenius & Q10 approximation)
        temp_delta = max(0.0, temp - 22.0)
        humid_delta = max(0.0, humidity - 60.0)
        spoilage_multiplier = round(1.0 + (temp_delta * 0.045) + (humid_delta * 0.015) + (0.15 if precip > 0.5 else 0.0), 2)
        reduction_pct = min(65, int(round((1.0 - (1.0 / spoilage_multiplier)) * 100)))

        if precip >= 0.5:
            condition = "Rain & Elevated Humidity"
            advisory = f"Active rain ({precip}mm/h) and moisture ({humidity}%) accelerate spoilage across bakery items, leafy greens, and soft fruits. Spoilage speed is +{int((spoilage_multiplier - 1.0)*100)}% above baseline."
        elif temp >= 33.0 and humidity >= 70.0:
            condition = "Extreme Heat & High Humidity"
            advisory = f"High ambient heat ({temp}°C) and moisture ({humidity}%) accelerate bacterial growth in Meat & Fish, Milk, and Curd. Spoilage speed is +{int((spoilage_multiplier - 1.0)*100)}% above baseline. Deep markdowns recommended."
        elif temp >= 32.0:
            condition = "Warm & Dry Climate"
            advisory = f"Elevated temperature ({temp}°C) accelerates moisture transpiration in Green Mango, Bananas, and Leafy Vegetables. Accelerated markdown recommended before weight shrinkage."
        elif humidity >= 78.0:
            condition = "High Ambient Moisture"
            advisory = f"Humid atmosphere ({humidity}%) promotes mold growth on bakery items and soft rot in tomatoes and bananas. Faster stock turnaround required."
        elif temp <= 25.0:
            condition = "Moderate & Controlled"
            advisory = f"Favorable ambient conditions ({temp}°C, {humidity}% humidity). Perishables maintain extended shelf-life retention with baseline discount schedules."
        else:
            condition = "Typical Retail Environment"
            advisory = f"Ambient conditions ({temp}°C, {humidity}% humidity) standard. PPO agent balancing margin vs clearance pace."

        return {
            "city": profile["city"],
            "state": profile["state"],
            "lat": profile.get("lat", 17.6868),
            "lon": profile.get("lon", 83.2185),
            "location_label": f"{profile['city']}, {profile['state']}",
            "temp": temp,
            "humidity": humidity,
            "precip": precip,
            "climate": climate,
            "condition": condition,
            "spoilage_multiplier": spoilage_multiplier,
            "shelf_life_reduction_pct": reduction_pct,
            "advisory": advisory,
            "is_hot": temp >= 31.0,
            "is_humid": humidity >= 75.0,
        }

    def get_supported_cities(self) -> List[Dict[str, Any]]:
        return [
            {"city": k, "state": v["state"], "climate": v["climate"]}
            for k, v in CITY_PROFILES.items()
        ]


weather_service = WeatherService()
