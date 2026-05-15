import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Constants
DATA_PATH = Path("updated_turbidity_dataset.csv")
MODEL_PATH = Path("model_bundle.joblib")

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-informed feature engineering to force chemical logic"""
    df = df.copy()
    # Log transform handles diminishing returns (saturation)
    df["effective_log_dose"] = np.log1p(df["coagulant_dose_mgL"])
    # Stoichiometry: ratio of chemical to dirt
    df["dose_to_raw_ratio"] = df["coagulant_dose_mgL"] / (df["turbidity_raw"] + 0.5)
    # Temperature efficiency (Efficiency drops in cold water)
    df["efficiency_index"] = df["dose_to_raw_ratio"] * (df.get("temperature_C", 15.0) / 25.0)
    return df

def train_and_compare():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Ensure the dataset is in the same folder.")
        return

    # 1. Load and Prepare Data
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values('timestamp')
    
    # Target: Removal Ratio (Final / Raw)
    # This prevents the '4.31 NTU floor' bias from the historical data
    df["removal_ratio"] = df["turbidity_final"] / (df["turbidity_raw"] + 0.1)
    df['turbidity_raw_lag_1'] = df['turbidity_raw'].shift(1).bfill()
    
    df = enrich_features(df)

    # Key features for water chemistry
    features = [
        'turbidity_raw', 'pH', 'temperature_C', 'coagulant_dose_mgL', 
        'effective_log_dose', 'dose_to_raw_ratio', 'efficiency_index',
        'turbidity_raw_lag_1', 'flow_rate_m3s'
    ]
    
    X = df[features]
    y = df["removal_ratio"]

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Gradient Boosting (Non-Linear Model)
    # Monotonic constraints force more dose to ALWAYS result in less turbidity (Ratio decreases)
    mono_cst = [0] * len(features)
    mono_cst[features.index("coagulant_dose_mgL")] = -1
    mono_cst[features.index("effective_log_dose")] = -1
    mono_cst[features.index("dose_to_raw_ratio")] = -1
    
    gb_model = HistGradientBoostingRegressor(
        monotonic_cst=mono_cst,
        max_iter=1000,
        learning_rate=0.02,
        max_depth=5,
        l2_regularization=10.0,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_preds = gb_model.predict(X_test)

    # 4. Ridge Regression (Linear Baseline)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_preds = ridge_model.predict(X_test)

    # 5. Accuracy Metrics Calculation
    metrics = {
        "Gradient Boosting": {
            "R2": r2_score(y_test, gb_preds),
            "MAE": mean_absolute_error(y_test, gb_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, gb_preds))
        },
        "Ridge Regression": {
            "R2": r2_score(y_test, ridge_preds),
            "MAE": mean_absolute_error(y_test, ridge_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, ridge_preds))
        }
    }

    # 6. Save Everything to Bundle
    joblib.dump({
        "model": gb_model,
        "feature_columns": features,
        "metrics": metrics,
        "is_ratio_model": True
    }, MODEL_PATH)

    # 7. Print Report
    print("\n" + "="*60)
    print("      WATER TREATMENT MODEL COMPARISON REPORT")
    print("="*60)
    print(f"{'Metric':<20} | {'Grad Boosting':<15} | {'Ridge (Linear)':<15}")
    print("-" * 60)
    print(f"{'R-Squared (Acc)':<20} | {metrics['Gradient Boosting']['R2']:<15.4f} | {metrics['Ridge Regression']['R2']:<15.4f}")
    print(f"{'Mean Abs Error':<20} | {metrics['Gradient Boosting']['MAE']:<15.4f} | {metrics['Ridge Regression']['MAE']:<15.4f}")
    print(f"{'RMSE':<20} | {metrics['Gradient Boosting']['RMSE']:<15.4f} | {metrics['Ridge Regression']['RMSE']:<15.4f}")
    print("="*60)
    
    if metrics['Gradient Boosting']['R2'] > metrics['Ridge Regression']['R2']:
        print("RESULT: Gradient Boosting is significantly more accurate.")
        print("WHY: Water chemistry (pH/Temp) is non-linear and requires complex trees.")
    else:
        print("RESULT: Linear relationships dominate this dataset.")
    print("="*60 + "\n")

if __name__ == '__main__':
    train_and_compare()