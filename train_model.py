"""
Used Car Price Prediction - Model Training Pipeline
Covers: Data Cleaning -> Feature Engineering -> Categorical Encoding
        -> Regression Model Training & Comparison -> Save Best Model
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_PATH = "/home/claude/car_price_prediction/data/used_cars.csv"
MODEL_DIR = "/home/claude/car_price_prediction/model"

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# ---------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------
# Fill missing numeric values with column median (robust to outliers)
for col in ["mileage_kmpl", "power_bhp", "seats"]:
    df[col] = df[col].fillna(df[col].median())

# ---------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------
CURRENT_YEAR = 2026

# a) car_age: more meaningful to a model than absolute manufacture year
df["car_age"] = CURRENT_YEAR - df["year"]

# b) km_driven_per_year: usage intensity, not just raw odometer reading
df["km_per_year"] = df["km_driven"] / df["car_age"].replace(0, 1)

# c) power_to_engine ratio: proxy for engine efficiency/performance
df["power_per_cc"] = df["power_bhp"] / df["engine_cc"]

# d) brand_avg_price: target-encoded style feature = brand's average price
#    (captures brand reputation/luxury tier that plain one-hot can't rank)
brand_avg = df.groupby("brand")["selling_price_lakhs"].mean()
df["brand_avg_price"] = df["brand"].map(brand_avg)

# e) is_luxury: engineered binary flag from domain knowledge
LUXURY_BRANDS = {"BMW", "Audi"}
df["is_luxury"] = df["brand"].isin(LUXURY_BRANDS).astype(int)

# f) log-transform skewed km_driven to reduce impact of outliers
df["km_driven_log"] = np.log1p(df["km_driven"])

print("\nEngineered features added: car_age, km_per_year, power_per_cc, "
      "brand_avg_price, is_luxury, km_driven_log")

# ---------------------------------------------------------
# 4. CATEGORICAL ENCODING
# ---------------------------------------------------------
categorical_cols = ["brand", "model", "fuel_type", "transmission",
                     "owner_type", "location"]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    label_encoders[col] = le

# ---------------------------------------------------------
# 5. FEATURE SET
# ---------------------------------------------------------
feature_cols = [
    "car_age", "km_driven", "km_driven_log", "km_per_year",
    "mileage_kmpl", "engine_cc", "power_bhp", "power_per_cc",
    "seats", "brand_avg_price", "is_luxury",
    "brand_enc", "model_enc", "fuel_type_enc",
    "transmission_enc", "owner_type_enc", "location_enc",
]

X = df[feature_cols]
y = df["selling_price_lakhs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 6. TRAIN & COMPARE REGRESSION MODELS
# ---------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
}

results = []
best_model = None
best_r2 = -np.inf
best_name = ""

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results.append({"Model": name, "R2 Score": round(r2, 4),
                     "MAE (Lakhs)": round(mae, 3), "RMSE (Lakhs)": round(rmse, 3)})

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))
print(f"\nBest Model: {best_name}  (R2 = {best_r2:.4f})")

# ---------------------------------------------------------
# 7. FEATURE IMPORTANCE (from best tree-based model if available)
# ---------------------------------------------------------
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print("\n===== FEATURE IMPORTANCE (Top 10) =====")
    print(importances.head(10).to_string())
    importances.to_csv(f"{MODEL_DIR}/feature_importance.csv")

# ---------------------------------------------------------
# 8. SAVE ARTIFACTS
# ---------------------------------------------------------
joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(label_encoders, f"{MODEL_DIR}/label_encoders.pkl")
joblib.dump(brand_avg.to_dict(), f"{MODEL_DIR}/brand_avg_price.pkl")

with open(f"{MODEL_DIR}/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

with open(f"{MODEL_DIR}/model_meta.json", "w") as f:
    json.dump({
        "best_model_name": best_name,
        "r2_score": round(best_r2, 4),
        "results": results,
        "brands": sorted(df["brand"].unique().tolist()),
        "models_by_brand": {b: sorted(df[df.brand == b]["model"].unique().tolist())
                             for b in df["brand"].unique()},
        "fuel_types": sorted(df["fuel_type"].unique().tolist()),
        "transmissions": sorted(df["transmission"].unique().tolist()),
        "owner_types": sorted(df["owner_type"].unique().tolist()),
        "locations": sorted(df["location"].unique().tolist()),
        "current_year": CURRENT_YEAR,
    }, f, indent=2)

results_df.to_csv(f"{MODEL_DIR}/model_comparison.csv", index=False)

print("\nAll artifacts saved to:", MODEL_DIR)
