import streamlit as st
import matplotlib.pyplot as plt
from optimize_coagulant_dose import min_dose_for_threshold
from predict_turbidity_final import load_model

st.set_page_config(page_title="💧 Dose Optimizer", layout="wide")
st.title("💧 Ratio-Based Coagulant Optimizer")

@st.cache_resource
def get_bundle():
    return load_model()

bundle = get_bundle()

st.sidebar.header("Water Quality")
rt_raw = st.sidebar.number_input("Raw Turbidity (NTU)", value=15.0)
rt_l1  = st.sidebar.number_input("Turbidity 1hr ago", value=14.5)
rt_ph  = st.sidebar.slider("pH", 6.0, 9.5, 7.2)
rt_temp = st.sidebar.slider("Temp (°C)", 4.0, 35.0, 15.0)
target = st.sidebar.slider("Target Final Turbidity", 0.1, 5.0, 1.0)

input_data = {
    "turbidity_raw": rt_raw,
    "pH": rt_ph,
    "temperature_C": rt_temp,
    "flow_rate_m3s": 1.5,
    "turbidity_raw_lag_1": rt_l1,
    "coagulant_dose_mgL": 0.0 # Placeholder
}

if st.button("Calculate Optimum Dose"):
    res = min_dose_for_threshold(input_data, target, bundle=bundle)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if res["feasible"]:
            st.success(f"Optimum Dose: {res['dose']:.2f} mg/L")
        else:
            st.error(f"Target Not Reachable. Best: {res['pred_at_dose']:.2f} NTU")
        st.metric("Predicted Final NTU", f"{res['pred_at_dose']:.2f}")

    with col2:
        fig, ax = plt.subplots()
        ax.plot(res["all_doses"], res["all_preds"], label="Dose Response", color='blue')
        ax.axhline(target, color='red', linestyle='--', label="Target")
        if res["feasible"]:
            ax.scatter(res["dose"], res["pred_at_dose"], color='green', s=100)
        ax.set_xlabel("Dose (mg/L)")
        ax.set_ylabel("Final Turbidity (NTU)")
        ax.legend()
        st.pyplot(fig)