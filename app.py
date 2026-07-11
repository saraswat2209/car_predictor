"""
Flask backend for the Used Car Price Prediction project.
Loads the trained model + encoders and exposes:
  GET  /            -> frontend UI
  GET  /api/options -> dropdown options (brands, models, fuel types etc.)
  POST /api/predict -> predicts price for given car features
"""
import json
import joblib
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR / "templates"),
    static_folder=str(FRONTEND_DIR / "static"),
)

# ---------------------------------------------------------
# Load trained artifacts once at startup
# ---------------------------------------------------------
model = joblib.load(MODEL_DIR / "best_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
label_encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")
brand_avg_price = joblib.load(MODEL_DIR / "brand_avg_price.pkl")

with open(MODEL_DIR / "feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

with open(MODEL_DIR / "model_meta.json") as f:
    META = json.load(f)

CURRENT_YEAR = META["current_year"]


def safe_label_transform(encoder, value, col_name):
    """Encode a category safely; fall back to most common class if unseen."""
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    # unseen category -> use the first known class as fallback
    return int(encoder.transform([encoder.classes_[0]])[0])


def build_feature_vector(payload):
    brand = payload["brand"]
    model_name = payload["model"]
    year = int(payload["year"])
    km_driven = float(payload["km_driven"])
    fuel_type = payload["fuel_type"]
    transmission = payload["transmission"]
    owner_type = payload["owner_type"]
    mileage = float(payload.get("mileage_kmpl", 18))
    engine_cc = float(payload.get("engine_cc", 1200))
    power_bhp = float(payload.get("power_bhp", 85))
    seats = float(payload.get("seats", 5))
    location = payload.get("location", META["locations"][0])

    # ----- feature engineering (must mirror train_model.py) -----
    car_age = max(0, CURRENT_YEAR - year)
    km_per_year = km_driven / (car_age if car_age else 1)
    power_per_cc = power_bhp / engine_cc if engine_cc else 0
    b_avg = brand_avg_price.get(brand, np.mean(list(brand_avg_price.values())))
    is_luxury = 1 if brand in ("BMW", "Audi") else 0
    km_driven_log = np.log1p(km_driven)

    row = {
        "car_age": car_age,
        "km_driven": km_driven,
        "km_driven_log": km_driven_log,
        "km_per_year": km_per_year,
        "mileage_kmpl": mileage,
        "engine_cc": engine_cc,
        "power_bhp": power_bhp,
        "power_per_cc": power_per_cc,
        "seats": seats,
        "brand_avg_price": b_avg,
        "is_luxury": is_luxury,
        "brand_enc": safe_label_transform(label_encoders["brand"], brand, "brand"),
        "model_enc": safe_label_transform(label_encoders["model"], model_name, "model"),
        "fuel_type_enc": safe_label_transform(label_encoders["fuel_type"], fuel_type, "fuel_type"),
        "transmission_enc": safe_label_transform(label_encoders["transmission"], transmission, "transmission"),
        "owner_type_enc": safe_label_transform(label_encoders["owner_type"], owner_type, "owner_type"),
        "location_enc": safe_label_transform(label_encoders["location"], location, "location"),
    }
    return np.array([[row[c] for c in FEATURE_COLS]])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/options")
def options():
    return jsonify({
        "brands": META["brands"],
        "models_by_brand": META["models_by_brand"],
        "fuel_types": META["fuel_types"],
        "transmissions": META["transmissions"],
        "owner_types": META["owner_types"],
        "locations": META["locations"],
        "current_year": CURRENT_YEAR,
        "model_name": META["best_model_name"],
        "r2_score": META["r2_score"],
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        X = build_feature_vector(payload)
        X_scaled = scaler.transform(X)
        pred_lakhs = float(model.predict(X_scaled)[0])
        pred_lakhs = max(0.3, pred_lakhs)

        return jsonify({
            "success": True,
            "predicted_price_lakhs": round(pred_lakhs, 2),
            "predicted_price_inr": int(round(pred_lakhs * 100000)),
            "model_used": META["best_model_name"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
