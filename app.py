import os
import json
import time
import re
import secrets
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, render_template, jsonify, request, redirect, url_for, session, Response
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv()

from src.model_service import model_service, resolve_store_id
from src.weather_service import weather_service
from src.perishable_classifier import classify_item
from src.email_service import email_service

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

# Generate a persistent secret key on first run, stored in data/secret.key
_secret_key_path = os.path.join(os.path.dirname(__file__), "data", "secret.key")
if os.path.exists(_secret_key_path):
    with open(_secret_key_path, "r") as _f:
        app.secret_key = _f.read().strip()
else:
    _generated_key = secrets.token_hex(32)
    os.makedirs(os.path.dirname(_secret_key_path), exist_ok=True)
    with open(_secret_key_path, "w") as _f:
        _f.write(_generated_key)
    app.secret_key = _generated_key


# ================================================================
#  CONTEXT PROCESSOR — injects live telemetry into every template
# ================================================================

def get_current_store_id() -> str:
    """Returns the isolated store ID for the current authenticated manager."""
    manager = session.get("manager")
    if not manager or not isinstance(manager, dict):
        return "flagship_kerala"
    store_id = manager.get("store_id")
    if not store_id:
        email = manager.get("email", "")
        store_id = resolve_store_id(email)
        manager["store_id"] = store_id
        session["manager"] = manager
        session.modified = True
    return store_id

@app.context_processor
def inject_globals():
    store_id = get_current_store_id() if "manager" in session else "flagship_kerala"
    suggestions = model_service.get_ppo_suggestions(store_id=store_id)
    expiry = model_service.get_expiry_items(store_id=store_id)
    store_state = model_service.get_store(store_id)
    settings = model_service.get_settings(store_id=store_id)
    manager = session.get("manager", {
        "name": "Store Manager",
        "designation": "General Store Manager",
        "location": settings.get("store_location", "Visakhapatnam"),
        "store_id": store_id,
    })
    
    # Retrieve persisted avatar_url if available
    avatar_url = manager.get("avatar_url", "")
    if not avatar_url and "email" in manager:
        try:
            mgr_file = BASE_DIR / "data" / "managers.json"
            if mgr_file.exists():
                with open(mgr_file, "r", encoding="utf-8") as f:
                    mgrs = json.load(f)
                for m in mgrs:
                    if m.get("email") == manager.get("email") and m.get("avatar_url"):
                        avatar_url = m["avatar_url"]
                        manager["avatar_url"] = avatar_url
                        session["manager"] = manager
                        break
        except Exception:
            pass

    weather = model_service.get_weather_telemetry(store_id=store_id)
    return {
        "store_id": store_id,
        "is_flagship": store_state.is_flagship,
        "store_name": settings.get("store_name", "EcoPrice Supermarket"),
        "active_items_count": len(store_state.catalog),
        "pending_suggestions": len([s for s in suggestions if s["status"] == "pending"]),
        "critical_count": sum(1 for e in expiry if e["urgency"] == "critical"),
        "warning_count": sum(1 for e in expiry if e["urgency"] == "warning"),
        "current_time": datetime.now().strftime("%I:%M %p"),
        "current_date": store_state.get_current_date().strftime("%A, %d %B %Y"),
        "sim_days_offset": int(store_state.settings.get("simulated_date_offset", 0)),
        "manager_name": manager.get("name", "Store Manager"),
        "manager_role": manager.get("designation", "General Store Manager"),
        "manager_location": manager.get("location", weather.get("location_label", "Visakhapatnam")),
        "manager_avatar": avatar_url,
        "weather": weather,
        "supported_cities": weather_service.get_supported_cities(),
    }


# ================================================================
#  AUTHENTICATION DECORATORS
# ================================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "manager" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def api_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "manager" not in session:
            return jsonify({"success": False, "error": "Authentication required. Please log in."}), 401
        return f(*args, **kwargs)
    return decorated_function


# ================================================================
#  PUBLIC & AUTHENTICATION ROUTES
# ================================================================

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "manager" in session:
        return redirect(url_for("dashboard"))
        
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        passcode = request.form.get("password", "")
        
        if not email or not passcode:
            return render_template("login.html", error="Please enter both identifier and passcode.")
            
        manager, status = model_service.authenticate(email, passcode)
        if manager:
            session.permanent = True
            session["manager"] = manager
            store_id = manager.get("store_id", "flagship_kerala")
            store = model_service.get_store(store_id)
            # The store's persistent saved location in settings.json is the ultimate source of truth!
            saved_store_loc = store.settings.get("store_location")
            if saved_store_loc:
                manager["location"] = saved_store_loc
                session["manager"]["location"] = saved_store_loc
                model_service.update_manager_location(email, saved_store_loc)
            elif "location" in manager and manager["location"]:
                model_service.set_location(manager["location"], store_id=store_id)
            return redirect(url_for("dashboard"))
        elif status == "unverified":
            return render_template(
                "login.html",
                error="Your store manager email has not been verified yet. Please check your inbox for the confirmation link or request a new one below.",
                unverified_email=email
            )
            
        return render_template("login.html", error="Invalid identifier or passcode.")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "manager" in session:
        return redirect(url_for("dashboard"))
        
    if request.method == "POST":
        name = request.form.get("name", "Store Manager").strip()
        email = request.form.get("email", "").strip()
        passcode = request.form.get("password", "")
        location = request.form.get("location", "Kochi")
        
        if not email or not passcode:
            return render_template(
                "signup.html", 
                error="Email identifier and passcode are required.",
                supported_cities=weather_service.get_supported_cities()
            )
            
        success, res = model_service.register_manager(
            email, passcode, designation=name, name=name, location=location, is_verified=False
        )
        if success:
            # Generate cryptographic confirmation token and dispatch verification email
            token = email_service.generate_token(email, app.secret_key)
            confirm_url = url_for("confirm_email", token=token, _external=True)
            is_sent, status_msg = email_service.send_confirmation_email(email, name, confirm_url)
            return render_template(
                "verify_pending.html",
                email=email,
                is_sent=is_sent,
                confirm_url=confirm_url
            )
        else:
            return render_template(
                "signup.html", 
                error=res, 
                supported_cities=weather_service.get_supported_cities()
            )
            
    return render_template("signup.html", supported_cities=weather_service.get_supported_cities())


@app.route("/confirm/<token>")
def confirm_email(token):
    email = email_service.verify_token(token, app.secret_key)
    if not email:
        return render_template(
            "verify_expired.html",
            error="This confirmation link is invalid or has expired (links expire after 24 hours). Please request a fresh link below."
        )

    success, manager = model_service.verify_manager(email)
    if success:
        # Automatically establish authenticated session for the verified manager
        session["manager"] = manager
        store_id = manager.get("store_id", "flagship_kerala")
        store = model_service.get_store(store_id)
        saved_store_loc = store.settings.get("store_location")
        if saved_store_loc:
            manager["location"] = saved_store_loc
            session["manager"]["location"] = saved_store_loc
        elif "location" in manager and manager["location"]:
            model_service.set_location(manager["location"], store_id=store_id)
        return render_template("verify_success.html", email=email, manager=manager)
    else:
        return render_template(
            "verify_expired.html",
            error="Unable to locate the registered account for verification."
        )


@app.route("/resend-confirmation", methods=["GET", "POST"])
def resend_confirmation():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        manager = model_service.get_manager_by_email(email)
        if manager:
            if manager.get("is_verified", False):
                return render_template("login.html", message="Your account is already verified. Please sign in.")
            token = email_service.generate_token(email, app.secret_key)
            confirm_url = url_for("confirm_email", token=token, _external=True)
            is_sent, status_msg = email_service.send_confirmation_email(
                email, manager.get("name", "Store Manager"), confirm_url
            )
            return render_template(
                "verify_pending.html",
                email=email,
                is_sent=is_sent,
                confirm_url=confirm_url,
                resent=True
            )
        return render_template(
            "resend_confirmation.html",
            error="No registered store account found with that email identifier."
        )
    email = request.args.get("email", "")
    return render_template("resend_confirmation.html", email=email)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))


# ================================================================
#  DASHBOARD & APPLICATION PAGES
# ================================================================

@app.route("/dashboard")
@login_required
def dashboard():
    store_id = get_current_store_id()
    suggestions = model_service.get_ppo_suggestions(store_id=store_id)
    expiry = model_service.get_expiry_items(store_id=store_id)
    store = model_service.get_store(store_id)
    return render_template(
        "dashboard.html",
        store_id=store_id,
        is_flagship=store.is_flagship,
        active_items_count=len(store.catalog),
        pending_suggestions=len([s for s in suggestions if s["status"] == "pending"]),
        critical_count=sum(1 for e in expiry if e["urgency"] == "critical"),
        current_time=datetime.now().strftime("%H:%M"),
        current_date=datetime.now().strftime("%A, %d %B %Y"),
        ai_summary=model_service.get_daily_ai_briefing(store_id=store_id),
        kpis=model_service.get_kpis(store_id=store_id),
        sales=model_service.get_sales_data(store_id=store_id),
        top_products=model_service.get_top_products(store_id=store_id),
        expiry_items=expiry,
    )


@app.route("/overview")
@login_required
def overview():
    store_id = get_current_store_id()
    store = model_service.get_store(store_id)
    return render_template(
        "overview.html",
        active_page="overview",
        store_id=store_id,
        is_flagship=store.is_flagship,
        active_items_count=len(store.catalog),
        kpis=model_service.get_kpis(store_id=store_id),
        alerts=model_service.get_alerts(store_id=store_id),
        top_products=model_service.get_top_products(store_id=store_id),
        sales=model_service.get_sales_data(store_id=store_id),
    )


@app.route("/sales")
@login_required
def sales():
    store_id = get_current_store_id()
    store = model_service.get_store(store_id)
    return render_template(
        "sales.html",
        active_page="sales",
        store_id=store_id,
        is_flagship=store.is_flagship,
        active_items_count=len(store.catalog),
        sales=model_service.get_sales_data(store_id=store_id),
        top_products=model_service.get_top_products(store_id=store_id),
        kpis=model_service.get_kpis(store_id=store_id),
    )


@app.route("/ppo")
@login_required
def ppo():
    store_id = get_current_store_id()
    store = model_service.get_store(store_id)
    all_suggestions = model_service.get_ppo_suggestions(store_id=store_id)
    pending_suggestions = [s for s in all_suggestions if s.get("status") == "pending"]
    resolved_suggestions = [s for s in all_suggestions if s.get("status") in ["accepted", "rejected"]]
    return render_template(
        "ppo.html",
        active_page="ppo",
        store_id=store_id,
        is_flagship=store.is_flagship,
        active_items_count=len(store.catalog),
        suggestions=pending_suggestions,
        resolved_suggestions=resolved_suggestions,
        all_suggestions=all_suggestions,
        history=model_service.get_discount_history(store_id=store_id),
        total_potential_savings=sum(
            (s["current_price"] - s["new_price"]) * s["stock"] for s in pending_suggestions
        ),
    )


@app.route("/forecast")
@login_required
def forecast():
    store_id = get_current_store_id()
    store = model_service.get_store(store_id)
    return render_template(
        "forecast.html",
        active_page="forecast",
        store_id=store_id,
        is_flagship=store.is_flagship,
        active_items_count=len(store.catalog),
        forecast=model_service.get_forecast_data(store_id=store_id),
    )


@app.route("/expiry")
@login_required
def expiry():
    store_id = get_current_store_id()
    store = model_service.get_store(store_id)
    items = model_service.get_expiry_items(store_id=store_id)
    return render_template(
        "expiry.html",
        active_page="expiry",
        store_id=store_id,
        is_flagship=store.is_flagship,
        active_items_count=len(store.catalog),
        items=items,
        total_at_risk=sum(e["value_at_risk"] for e in items),
        critical_count_page=sum(1 for e in items if e["urgency"] == "critical"),
        warning_count=sum(1 for e in items if e["urgency"] == "warning"),
    )


@app.route("/settings")
@login_required
def settings():
    store_id = get_current_store_id()
    store = model_service.get_store(store_id)
    return render_template(
        "settings.html",
        active_page="settings",
        store_id=store_id,
        is_flagship=store.is_flagship,
        active_items_count=len(store.catalog),
        settings=model_service.get_settings(store_id=store_id),
        inventory_items=model_service.get_expiry_items(store_id=store_id),
        catalog=store.catalog,
        sales_records=list(reversed(store.sales_records))
    )


# ================================================================
#  REST API ENDPOINTS
# ================================================================

@app.route("/api/suggestion/<int:sid>/<action>", methods=["POST"])
@api_login_required
def update_suggestion(sid, action):
    """Accept, reject, or undo an individual PPO discount suggestion."""
    store_id = get_current_store_id()
    action_lower = action.lower()
    if action_lower in ["undo", "reset", "revert"]:
        status = "pending"
    elif action_lower in ["accept", "accepted", "approve", "approved", "apply", "confirm"]:
        status = "accepted"
    else:
        status = "rejected"
        
    model_service.update_suggestion_status(sid, status, store_id=store_id)
    item = next((s for s in model_service.get_ppo_suggestions(store_id=store_id) if s["id"] == sid), None)
    return jsonify({
        "success": True,
        "id": sid,
        "action": status,
        "item": item,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/bulk-action", methods=["POST"])
@api_login_required
def bulk_action():
    """Bulk accept or reject all suggestions."""
    store_id = get_current_store_id()
    data = request.get_json() or {}
    action = data.get("action", "accept-all")
    resulting_status = model_service.bulk_update_suggestions(action, store_id=store_id)
    return jsonify({
        "success": True,
        "action": action,
        "status": resulting_status
    })


@app.route("/api/save-settings", methods=["POST"])
@api_login_required
def save_settings():
    """Saves updated model penalty hyperparameters and shelf-life constraints."""
    store_id = get_current_store_id()
    data = request.get_json() or {}
    updated = model_service.update_settings(data, store_id=store_id)
    return jsonify({
        "success": True,
        "settings": updated,
        "message": "Parameters updated and synced to PPO inference engine."
    })


@app.route("/api/upload-dataset", methods=["POST"])
@api_login_required
def upload_dataset():
    """Handles manager upload of new sales CSV or Excel logs."""
    store_id = get_current_store_id()
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file stream provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Empty file name."}), 400

    filename = secure_filename(file.filename) or "upload.csv"
    res = model_service.process_uploaded_dataset(filename, file, store_id=store_id)
    return jsonify(res)


@app.route("/api/upload-avatar", methods=["POST"])
@api_login_required
def upload_avatar():
    """Uploads a manager profile picture (strictly under 3MB) and stores it."""
    if "avatar" not in request.files:
        return jsonify({"success": False, "error": "No image file provided."}), 400

    file = request.files["avatar"]
    if not file or not file.filename:
        return jsonify({"success": False, "error": "No file selected."}), 400

    orig_name = file.filename.lower()
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}
    ext = os.path.splitext(orig_name)[1]
    if ext not in allowed_exts:
        return jsonify({"success": False, "error": f"Invalid format '{ext}'. Allowed: JPG, PNG, WEBP."}), 400

    MAX_BYTES = 3 * 1024 * 1024  # 3 MB
    content = file.read()
    if len(content) > MAX_BYTES:
        return jsonify({
            "success": False,
            "error": f"Image size ({len(content)/(1024*1024):.2f}MB) exceeds the maximum allowed limit of 3MB."
        }), 400

    if len(content) == 0:
        return jsonify({"success": False, "error": "Uploaded image file is empty."}), 400

    # Ensure directory
    avatar_dir = BASE_DIR / "static" / "uploads" / "avatars"
    avatar_dir.mkdir(parents=True, exist_ok=True)

    mgr_email = session.get("manager", {}).get("email", "manager")
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", mgr_email)
    out_filename = f"{safe_name}_{int(time.time())}{ext}"
    out_path = avatar_dir / out_filename

    with open(out_path, "wb") as f:
        f.write(content)

    avatar_url = f"/static/uploads/avatars/{out_filename}"

    # Update session
    if "manager" in session:
        session["manager"]["avatar_url"] = avatar_url
        session.modified = True

    # Persist into managers.json
    try:
        mgr_file = BASE_DIR / "data" / "managers.json"
        if mgr_file.exists():
            with open(mgr_file, "r", encoding="utf-8") as f:
                mgrs = json.load(f)
            for m in mgrs:
                if m.get("email") == mgr_email:
                    m["avatar_url"] = avatar_url
                    break
            with open(mgr_file, "w", encoding="utf-8") as f:
                json.dump(mgrs, f, indent=4)
    except Exception as e:
        print(f"[UploadAvatar] Error writing managers.json: {e}")

    return jsonify({
        "success": True,
        "avatar_url": avatar_url,
        "message": "Manager profile picture updated successfully."
    })


@app.route("/api/calendar/advance", methods=["POST"])
@api_login_required
def advance_calendar():
    """Advances the active store calendar forward by N days to simulate time-travel and dynamic countdown."""
    store_id = get_current_store_id()
    data = request.get_json() or {}
    days = int(data.get("days", 1))
    res = model_service.advance_date(days=days, store_id=store_id)
    return jsonify(res)


@app.route("/api/calendar/reset", methods=["POST"])
@api_login_required
def reset_calendar():
    """Resets the active store calendar back to today's real date."""
    store_id = get_current_store_id()
    res = model_service.reset_date(store_id=store_id)
    return jsonify(res)


@app.route("/api/update-location", methods=["POST"])
@api_login_required
def update_location():
    """Updates the active store location and recalibrates weather-driven spoilage and PPO markdowns."""
    store_id = get_current_store_id()
    data = request.get_json() or {}
    loc = data.get("location", "Visakhapatnam")
    weather = model_service.set_location(loc, store_id=store_id)
    clean_city = weather.get("city", loc)
    if "manager" in session:
        session["manager"]["location"] = clean_city
        session.modified = True
    mgr_email = session.get("manager", {}).get("email")
    if mgr_email:
        model_service.update_manager_location(mgr_email, clean_city)
    return jsonify({
        "success": True,
        "location": weather["location_label"],
        "weather": weather,
        "message": f"Store location updated to {weather['location_label']}. Spoilage acceleration index: {weather['spoilage_multiplier']}x."
    })


@app.route("/api/detect-location", methods=["POST"])
@api_login_required
def detect_location():
    """Detects exact store locality and live weather via browser GPS coordinates and reverse geocoding."""
    store_id = get_current_store_id()
    data = request.get_json() or {}
    try:
        lat = float(data.get("lat", 9.9312))
        lon = float(data.get("lon", 76.2673))
        suggested_city = data.get("city", "").strip()
        suggested_state = data.get("state", "").strip()
        detected_city, detected_state, weather = weather_service.register_detected_gps_location(
            lat=lat,
            lon=lon,
            suggested_city=suggested_city,
            suggested_state=suggested_state
        )
    except Exception as e:
        detected_city = "Visakhapatnam"
        detected_state = "Andhra Pradesh"
        weather = weather_service.get_weather("Visakhapatnam")

    model_service.set_location(detected_city, store_id=store_id)
    if "manager" in session:
        session["manager"]["location"] = detected_city
        session.modified = True
    mgr_email = session.get("manager", {}).get("email")
    if mgr_email:
        model_service.update_manager_location(mgr_email, detected_city)
    return jsonify({
        "success": True,
        "city": detected_city,
        "state": detected_state,
        "location": weather["location_label"],
        "weather": weather,
        "message": f"Detected location: {weather['location_label']} ({weather['temp']}°C, {weather['humidity']}% RH)."
    })


@app.route("/api/add-product", methods=["POST"])
@api_login_required
def add_product():
    """Dynamically registers a brand new SKU or delivery batch into store inventory."""
    data = request.get_json() or {}
    product_name = (data.get("product") or data.get("name") or "").strip()
    category = (data.get("category") or "Specialty & Fresh").strip()
    try:
        raw_rate = data.get("rate") if data.get("rate") is not None else data.get("price", 100.0)
        rate = float(raw_rate)
        raw_shelf = data.get("shelf_life") if data.get("shelf_life") is not None else data.get("shelf_life_days", 2)
        shelf_life = max(1, int(raw_shelf))
        stock = max(1, int(data.get("stock", 20)))
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Invalid numerical parameters"}), 400

    if not product_name:
        return jsonify({"success": False, "error": "Product name is required"}), 400

    # Perishable classification gate: reject non-perishable items
    is_perishable, reason, confidence = classify_item(
        product_name=product_name,
        category=category,
        shelf_life_days=shelf_life
    )
    if not is_perishable:
        return jsonify({
            "success": False,
            "error": f"'{product_name}' was identified as a non-perishable item and cannot be added to the shelf-life monitoring system. Reason: {reason}"
        }), 400

    store_id = get_current_store_id()
    expiry_date = data.get("expiry_date") or None
    received_date = data.get("received_date") or None
    item = model_service.add_product_or_batch(
        product_name=product_name,
        category=category,
        rate=rate,
        shelf_life=shelf_life,
        stock=stock,
        received_date=received_date,
        expiry_date=expiry_date,
        store_id=store_id
    )
    return jsonify({
        "success": True,
        "item": item,
        "message": f"Successfully registered {item['product']} ({item['stock']} units, batch {item['batch']}) into live inventory."
    })


@app.route("/api/load-starter-pack", methods=["POST"])
@api_login_required
def load_starter_pack():
    """Loads 8 pre-configured common grocery perishables for immediate onboarding testing."""
    store_id = get_current_store_id()
    items = model_service.load_starter_pack(store_id=store_id)
    return jsonify({
        "success": True,
        "count": len(items),
        "items": items,
        "message": f"Successfully loaded sample starter pack ({len(items)} items)."
    })


@app.route("/api/retrain", methods=["POST"])
@api_login_required
def retrain():
    """Triggers fine-tuning and recalibration of XGBoost & PPO models on newly uploaded data."""
    res = model_service.retrain_models()
    return jsonify(res)


@app.route("/api/reset-dataset", methods=["POST"])
@api_login_required
def reset_dataset():
    """Rolls back the sales register database and inventory caches to clean factory baseline."""
    store_id = get_current_store_id()
    res = model_service.reset_to_golden_dataset(store_id=store_id)
    return jsonify(res)


@app.route("/api/update-stock", methods=["POST"])
@api_login_required
def update_stock():
    """Updates stock count for a product after daily sales or stock-take."""
    store_id = get_current_store_id()
    data = request.get_json() or {}
    product_id = data.get("id")
    new_stock = data.get("stock")
    sold_qty = data.get("sold")

    if product_id is None:
        return jsonify({"success": False, "error": "Product id is required"}), 400

    try:
        product_id = int(product_id)
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Invalid product id"}), 400

    result = model_service.update_stock(product_id, new_stock=new_stock, sold_qty=sold_qty, store_id=store_id)
    if result:
        return jsonify({"success": True, "item": result, "message": f"Stock updated for {result['product']}: {result['stock']} units remaining."})
    return jsonify({"success": False, "error": "Product not found"}), 404


@app.route("/api/bulk-update-sales", methods=["POST"])
@api_login_required
def bulk_update_sales():
    """Reconciles end-of-day sales across multiple products in bulk."""
    store_id = get_current_store_id()
    data = request.get_json() or {}
    updates = data.get("updates", [])
    if not updates and "id" in data:
        updates = [data]
    res = model_service.bulk_update_stock(updates, store_id=store_id)
    return jsonify(res)


@app.route("/api/delete-product", methods=["POST"])
@api_login_required
def delete_product():
    """Removes a product/SKU from the active perishable registry and inventory."""
    store_id = get_current_store_id()
    data = request.get_json() or {}
    product_id = data.get("id")
    product_name = data.get("name")

    if product_id is None and not product_name:
        return jsonify({"success": False, "error": "Product id or name is required"}), 400

    deleted_item = model_service.delete_product(product_id if product_id is not None else product_name, store_id=store_id)
    if not deleted_item:
        return jsonify({"success": False, "error": "Product not found in registry"}), 404

    return jsonify({
        "success": True,
        "product": deleted_item["product"],
        "id": deleted_item["id"],
        "message": f"Successfully removed '{deleted_item['product']}' from registry and monitoring."
    })


@app.route("/api/registry/delete/<int:product_id>", methods=["POST"])
@api_login_required
def delete_registry_product(product_id):
    """Deletes a product SKU from the registry by its ID."""
    store_id = get_current_store_id()
    deleted_item = model_service.delete_product(product_id, store_id=store_id)
    if not deleted_item:
        return jsonify({"success": False, "error": "Product not found in registry"}), 404

    return jsonify({
        "success": True,
        "product": deleted_item["product"],
        "id": deleted_item["id"],
        "message": f"Successfully removed '{deleted_item['product']}' from registry and monitoring."
    })


@app.route("/api/export-decision-history")
@login_required
def export_decision_history():
    """Exports the entire permanent decision history audit log as a downloadable CSV spreadsheet."""
    import csv, io
    from flask import Response

    store_id = get_current_store_id()
    history = model_service.get_discount_history(store_id=store_id)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Decision ID", "Date", "Time", "Product", "Category",
        "Original Price (Rs)", "Discount (%)", "New Price (Rs)", "Action",
        "Stock", "Result", "Revenue Impact", "Store Location", "Timestamp"
    ])

    for h in history:
        writer.writerow([
            h.get("id") or f"DEC-{abs(hash(str(h.get('timestamp', '')))) % 1000000:06d}",
            h.get("date", ""),
            h.get("time", ""),
            h.get("product", ""),
            h.get("category", ""),
            h.get("original_price", ""),
            h.get("discount", h.get("discount_pct", "")),
            h.get("new_price", ""),
            h.get("action", ""),
            h.get("stock", ""),
            h.get("result", ""),
            h.get("revenue_impact", ""),
            h.get("location", ""),
            h.get("iso_timestamp", "")
        ])

    output.seek(0)
    filename = f"ecoprice_decision_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/api/export-sales-records")
@login_required
def export_sales_records():
    """Streams full transaction log of committed sales for this store as downloadable CSV."""
    import io
    import csv
    store_id = get_current_store_id()
    store = model_service.get_store(store_id)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Transaction ID",
        "Date",
        "Timestamp",
        "Product",
        "Category",
        "Units Sold",
        "Unit Price (Rs)",
        "Total Revenue (Rs)",
        "Store Location"
    ])
    records = sorted(store.sales_records, key=lambda x: str(x.get("timestamp", "")), reverse=True)
    for idx, r in enumerate(records):
        rid = r.get("id") or f"TXN-{idx+1:05d}"
        writer.writerow([
            rid,
            r.get("date", ""),
            r.get("timestamp", ""),
            r.get("product", ""),
            r.get("category", ""),
            r.get("units_sold", 0),
            r.get("price", 0.0),
            r.get("total_revenue", 0.0),
            store.settings.get("store_location", "Visakhapatnam")
        ])
    output.seek(0)
    filename = f"ecoprice_sales_register_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/api/sales-record/<record_id>/update", methods=["POST"])
@api_login_required
def update_sales_record(record_id):
    """Updates units sold in a committed sales record and dynamically adjusts inventory stock."""
    store_id = get_current_store_id()
    data = request.get_json() or {}
    new_units = data.get("units_sold")
    if new_units is None:
        return jsonify({"success": False, "error": "Missing units_sold parameter."}), 400
    try:
        new_units = int(new_units)
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Invalid units_sold value."}), 400

    res = model_service.edit_sales_record(record_id, new_units, store_id=store_id)
    return jsonify(res)


@app.route("/api/sales-record/<record_id>/delete", methods=["POST"])
@api_login_required
def delete_sales_record_route(record_id):
    """Deletes a mistakenly committed sales record and refunds all sold units back to inventory stock."""
    store_id = get_current_store_id()
    res = model_service.delete_sales_record(record_id, store_id=store_id)
    return jsonify(res)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ["1", "true"]
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
