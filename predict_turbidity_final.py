import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor

MODEL_PATH = Path("model_bundle.joblib")
DATA_PATH = Path("updated_turbidity_dataset.csv")

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Physics features to force the model to see dose impact
    df["effective_log_dose"] = np.log1p(df["coagulant_dose_mgL"])
    df["dose_to_raw_ratio"] = df["coagulant_dose_mgL"] / (df["turbidity_raw"] + 0.5)
    df["efficiency_index"] = df["dose_to_raw_ratio"] * (df.get("temperature_C", 15.0) / 25.0)
    return df

def train():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    df = df.sort_values('timestamp')
    df['turbidity_raw_lag_1'] = df['turbidity_raw'].shift(1).bfill()
    
    # THE RATIO FIX: Target is now percentage of turbidity left
    # This prevents the model from being stuck at the 4.31 NTU average
    df["removal_ratio"] = df["turbidity_final"] / (df["turbidity_raw"] + 0.1)
    
    df = enrich_features(df)

    features = [
        'turbidity_raw', 'pH', 'temperature_C', 'coagulant_dose_mgL', 
        'effective_log_dose', 'dose_to_raw_ratio', 'efficiency_index',
        'turbidity_raw_lag_1', 'flow_rate_m3s'
    ]
    
    target = "removal_ratio"
    X = df[features]
    y = df[target]

    # Constraints to ensure more dose = lower ratio (more removal)
    mono_cst = [0] * len(features)
    mono_cst[features.index("coagulant_dose_mgL")] = -1
    mono_cst[features.index("effective_log_dose")] = -1
    mono_cst[features.index("dose_to_raw_ratio")] = -1
    mono_cst[features.index("efficiency_index")] = -1

    model = HistGradientBoostingRegressor(
        monotonic_cst=mono_cst,
        max_iter=1500,
        learning_rate=0.01,
        max_depth=3, # Shorter trees generalize better
        l2_regularization=50.0,
        random_state=42
    )

    model.fit(X, y)

    joblib.dump({
        "model": model, 
        "feature_columns": features,
        "is_ratio_model": True 
    }, MODEL_PATH)
    print("✅ Ratio-Based Model Trained. Target Bias Removed.")

def load_model():
    return joblib.load(MODEL_PATH)

if __name__ == "__main__":
    train()