import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------- CONFIG ----------------
CSV_PATH = "updated_turbidity_dataset.csv"
MODEL_PATH = "ridge_model_bundle.pkl"

FEATURES = [
    "turbidity_raw", "pH", "alkalinity_mgL", "hardness_mgL", "temperature_C",
    "flow_rate_m3s", "coagulant_dose_mgL", "turbidity_after_coagulation",
    "turbidity_after_sedimentation", "turbidity_after_filtration",
    "rainfall_intensity_mm_hr", "TSS_mgL", "DOC_mgL", "conductivity_uScm",
    "mixing_intensity_G_s", "flocculation_time_min",
]

TARGET = "turbidity_final"


# ---------------- LOAD DATA ----------------
df = pd.read_csv(CSV_PATH).dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]


# ---------------- PIPELINE ----------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

pipeline.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = pipeline.predict(X_test)

print("\n--- Ridge Model Performance ---")
print("R2   :", r2_score(y_test, y_pred))
print("MAE  :", mean_absolute_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))


# ---------------- SAVE MODEL ----------------
bundle = {
    "model": pipeline,
    "feature_columns": FEATURES,
    "use_engineered": False
}

joblib.dump(bundle, MODEL_PATH)
print(f"\nSaved model → {MODEL_PATH}")