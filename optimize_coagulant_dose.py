import numpy as np
import pandas as pd
from predict_turbidity_final import load_model, enrich_features

def min_dose_for_threshold(
    sample: dict, 
    threshold_final: float, 
    bundle: dict = None, 
    dose_min: float = 0.0, 
    dose_max: float = 300, 
    n_grid: int = 300
):
    b = bundle or load_model()
    doses = np.linspace(dose_min, dose_max, n_grid)
    
    temp_df = pd.DataFrame([sample] * n_grid)
    temp_df["coagulant_dose_mgL"] = doses
    temp_df = enrich_features(temp_df)
    
    # Predict the RATIO
    preds_ratio = b["model"].predict(temp_df[b["feature_columns"]])
    
    # Convert ratio back to absolute NTU: Final = Ratio * Raw
    preds_final = preds_ratio * (sample["turbidity_raw"] + 0.1)
    
    feasible_idx = np.where(preds_final <= threshold_final)[0]
    
    if len(feasible_idx) > 0:
        idx = feasible_idx[0]
        return {
            "dose": float(doses[idx]), 
            "pred_at_dose": float(preds_final[idx]), 
            "feasible": True,
            "all_preds": preds_final,
            "all_doses": doses
        }
    
    return {
        "dose": float(doses[-1]), 
        "pred_at_dose": float(preds_final[-1]), 
        "feasible": False,
        "all_preds": preds_final,
        "all_doses": doses
    }