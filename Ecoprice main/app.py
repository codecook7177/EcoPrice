import os
import json
import time
import re
import secrets
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv()

from src.model_service import model_service
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

@app.context_processor
def inject_globals():
    suggestions = model_service.get_ppo_suggestions()
    expiry = model_service.get_expiry_items()
    manager = session.get("manager", {
        "name": "Store Manager",
        "designation": "General Store Manager",
        "location": model_service.current_location,
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

    weather = model_service.get_weather_telemetry()
    return {
        "pending_suggestions": len([s for s in suggestions if s["status"] == "pending"]),
        "critical_count": sum(1 for e in expiry if e["urgency"] == "critical"),
        "warning_count": sum(1 for e in expiry if e["urgency"] == "warning"),
        "current_time": datetime.now().strftime("%I:%M %p"),
        "current_date": datetime.now().strftime("%A, %d %B %Y"),
        "manager_name": manager.get("name", "Store Manager"),
        "manager_role": manager.get("designation", "General Store Manager"),
        "manager_location": manager.get("location", weather.get("location_label", "Kochi, Kerala")),
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
            if "location" in manager:
                model_service.set_location(manager["location"])
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
        if "location" in manager:
            model_service.set_location(manager["location"])
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
    suggestions = model_service.get_ppo_suggestions()
    expiry = model_service.get_expiry_items()
    return render_template(
        "dashboard.html",
        pending_suggestions=len([s for s in suggestions if s["status"] == "pending"]),
        critical_count=sum(1 for e in expiry if e["urgency"] == "critical"),
        current_time=datetime.now().strftime("%H:%M"),
        current_date=datetime.now().strftime("%A, %d %B %Y"),
        ai_summary=model_service.get_daily_ai_briefing(),
        kpis=model_service.get_kpis(),
        sales=model_service.get_sales_data(),
        top_products=model_service.get_top_products(),
    )


@app.route("/overview")
@login_required
def overview():
    return render_template(
        "overview.html",
        active_page="overview",
        kpis=model_service.get_kpis(),
        alerts=model_service.get_alerts(),
        top_products=model_service.get_top_products(),
        sales=model_service.get_sales_data(),
    )


@app.route("/sales")
@login_required
def sales():
    return render_template(
        "sales.html",
        active_page="sales",
        sales=model_service.get_sales_data(),
        top_products=model_service.get_top_products(),
        kpis=model_service.get_kpis(),
    )


@app.route("/ppo")
@login_required
def ppo():
    all_suggestions = model_service.get_ppo_suggestions()
    pending_suggestions = [s for s in all_suggestions if s.get("status") == "pending"]
    resolved_suggestions = [s for s in all_suggestions if s.get("status") in ["accepted", "rejected"]]
    return render_template(
        "ppo.html",
        active_page="ppo",
        suggestions=pending_suggestions,
        resolved_suggestions=resolved_suggestions,
        all_suggestions=all_suggestions,
        history=model_service.get_discount_history(),
        total_potential_savings=sum(
            (s["current_price"] - s["new_price"]) * s["stock"] for s in pending_suggestions
        ),
    )


@app.route("/forecast")
@login_required
def forecast():
    return render_template(
        "forecast.html",
        active_page="forecast",
        forecast=model_service.get_forecast_data(),
    )


@app.route("/expiry")
@login_required
def expiry():
    items = model_service.get_expiry_items()
    return render_template(
        "expiry.html",
        active_page="expiry",
        items=items,
        total_at_risk=sum(e["value_at_risk"] for e in items),
        critical_count_page=sum(1 for e in items if e["urgency"] == "critical"),
        warning_count=sum(1 for e in items if e["urgency"] == "warning"),
    )


@app.route("/settings")
@login_required
def settings():
    return render_template(
        "settings.html",
        active_page="settings",
        settings=model_service.get_settings(),
        inventory_items=model_service.get_expiry_items()
    )


# ================================================================
#  REST API ENDPOINTS
# ================================================================

@app.route("/api/suggestion/<int:sid>/<action>", methods=["POST"])
@api_login_required
def update_suggestion(sid, action):
    """Accept, reject, or undo an individual PPO discount suggestion."""
    action_lower = action.lower()
    if action_lower in ["undo", "reset", "revert"]:
        status = "pending"
    elif action_lower in ["accept", "accepted", "approve", "approved", "apply", "confirm"]:
        status = "accepted"
    else:
        status = "rejected"
        
    model_service.update_suggestion_status(sid, status)
    item = next((s for s in model_service.get_ppo_suggestions() if s["id"] == sid), None)
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
    data = request.get_json() or {}
    action = data.get("action", "accept-all")
    resulting_status = model_service.bulk_update_suggestions(action)
    return jsonify({
        "success": True,
        "action": action,
        "status": resulting_status
    })


@app.route("/api/save-settings", methods=["POST"])
@api_login_required
def save_settings():
    """Saves updated model penalty hyperparameters and shelf-life constraints."""
    data = request.get_json() or {}
    updated = model_service.update_settings(data)
    return jsonify({
        "success": True,
        "settings": updated,
        "message": "Parameters updated and synced to PPO inference engine."
    })


@app.route("/api/upload-dataset", methods=["POST"])
@api_login_required
def upload_dataset():
    """Handles manager upload of new sales CSV or Excel logs."""
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file stream provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Empty file name."}), 400

    filename = secure_filename(file.filename) or "upload.csv"
    res = model_service.process_uploaded_dataset(filename, file)
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


@app.route("/api/update-location", methods=["POST"])
@api_login_required
def update_location():
    """Updates the active store location and recalibrates weather-driven spoilage and PPO markdowns."""
    data = request.get_json() or {}
    loc = data.get("location", "Kochi")
    weather = model_service.set_location(loc)
    if "manager" in session:
        session["manager"]["location"] = weather["location_label"]
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
        detected_city = "Kochi"
        detected_state = "Kerala"
        weather = weather_service.get_weather("Kochi")

    model_service.set_location(detected_city)
    if "manager" in session:
        session["manager"]["location"] = weather["location_label"]
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

    item = model_service.add_product_or_batch(
        product_name=product_name,
        category=category,
        rate=rate,
        shelf_life=shelf_life,
        stock=stock
    )
    return jsonify({
        "success": True,
        "item": item,
        "message": f"Successfully registered {item['product']} ({item['stock']} units, batch {item['batch']}) into live inventory."
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
    res = model_service.reset_to_golden_dataset()
    return jsonify(res)


@app.route("/api/update-stock", methods=["POST"])
@api_login_required
def update_stock():
    """Updates stock count for a product after daily sales or stock-take."""
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

    result = model_service.update_stock(product_id, new_stock=new_stock, sold_qty=sold_qty)
    if result:
        return jsonify({"success": True, "item": result, "message": f"Stock updated for {result['product']}: {result['stock']} units remaining."})
    return jsonify({"success": False, "error": "Product not found"}), 404


@app.route("/api/bulk-update-sales", methods=["POST"])
@api_login_required
def bulk_update_sales():
    """Reconciles end-of-day sales across multiple products in bulk."""
    data = request.get_json() or {}
    updates = data.get("updates", [])
    if not updates and "id" in data:
        updates = [data]
    res = model_service.bulk_update_stock(updates)
    return jsonify(res)


@app.route("/api/delete-product", methods=["POST"])
@api_login_required
def delete_product():
    """Removes a product/SKU from the active perishable registry and inventory."""
    data = request.get_json() or {}
    product_id = data.get("id")
    product_name = data.get("name")

    if product_id is None and not product_name:
        return jsonify({"success": False, "error": "Product id or name is required"}), 400

    deleted_item = model_service.delete_product(product_id if product_id is not None else product_name)
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
    deleted_item = model_service.delete_product(product_id)
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

    history = model_service.get_discount_history()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Decision ID", "Date", "Time", "Product", "Category",
        "Original Price (Rs)", "Discount (%)", "New Price (Rs)", "Action",
        "Stock", "Result", "Revenue Impact", "Store Location", "Timestamp"
    ])

    for h in history:
        writer.writerow([
            h.get("id", f"DEC-{int(h.get('timestamp', 0))}"),
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ["1", "true"]
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
