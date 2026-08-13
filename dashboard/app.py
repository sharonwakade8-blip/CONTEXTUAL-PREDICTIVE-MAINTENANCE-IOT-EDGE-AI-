import sys
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

# Ensure Src/ is in the python path so we can import predict.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Src')))
from predict import PredictiveMaintenanceInference

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IoT Edge AI | Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-title { font-size: 1.1rem; color: #555; }
    .metric-value { font-size: 2.2rem; font-weight: bold; color: #1f77b4; }
    .status-normal { color: #2ca02c; font-weight: bold; }
    .status-warning { color: #ff7f0e; font-weight: bold; }
    .status-critical { color: #d62728; font-weight: bold; }
    
    .stProgress > div > div > div > div {
        background-color: #d62728;
    }
</style>
""", unsafe_allow_html=True)


# --- CACHE INFERENCE MODEL ---
@st.cache_resource
def load_inference_engine_v2():
    # Looks for Models/ in the parent directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'Models', 'lightgbm_model.pkl')
    scaler_path = os.path.join(base_dir, 'Models', 'scaler.pkl')
    
    try:
        return PredictiveMaintenanceInference(model_path=model_path, scaler_path=scaler_path)
    except FileNotFoundError:
        st.error(f"❌ Model files not found! Expected at:\n- {model_path}\n- {scaler_path}")
        st.stop()


inference_engine = load_inference_engine_v2()


# --- SIDEBAR NAV ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2165/2165686.png", width=100)
st.sidebar.title("IoT Edge Control")
page = st.sidebar.radio("Navigation", [
    "🏭 Fleet Overview",
    "🎛️ What-If Simulation",
    "🧠 SHAP Explainability",
    "⚙️ Threshold & Noise Tuning"
])

st.sidebar.markdown("---")

# --- PAGE 1: FLEET OVERVIEW ---
if page == "🏭 Fleet Overview":
    st.title("🏭 Plant Fleet Health Overview")
    st.markdown("Monitor real-time sensor streams and identify machines with a **>80% failure risk** within the next 7 days.")
    
    # Load batch data for the fleet view
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'Data', 'raw', 'ai4i2020.csv')
    
    try:
        raw_df = pd.read_csv(data_path)
        # Sample 100 rows to simulate active fleet
        fleet_df = raw_df.sample(100, random_state=42).reset_index(drop=True) 
        
        # We need to map Type to numeric as done in preprocessing
        type_mapping = {'L': 1, 'M': 2, 'H': 0} # common label encoding mapping
        fleet_df['Type_Encoded'] = fleet_df['Type'].map(type_mapping).fillna(1)
        
        # Prepare for batch predict (rename to match expected)
        batch_input = fleet_df.copy()
        batch_input['Type'] = batch_input['Type_Encoded']
        
        results_df = inference_engine.predict_batch(batch_input)
        
        # KPIs
        total_machines = len(results_df)
        critical = len(results_df[results_df['Status'] == 'CRITICAL'])
        warning = len(results_df[results_df['Status'] == 'WARNING'])
        normal = total_machines - critical - warning
        
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f'<div class="metric-card"><div class="metric-title">Total Active</div><div class="metric-value">{total_machines}</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><div class="metric-title">Normal Status</div><div class="metric-value" style="color: #2ca02c;">{normal}</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><div class="metric-title">Warnings</div><div class="metric-value" style="color: #ff7f0e;">{warning}</div></div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="metric-card"><div class="metric-title">Critical Risk</div><div class="metric-value" style="color: #d62728;">{critical}</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display Critical Alerts
        if critical > 0:
            st.error(f"🚨 **ALERT:** {critical} machine(s) have a failure probability > 80%. Immediate inspection required!")
        
        # Data Table
        st.subheader("📡 Live Telemetry & Predictions")
        display_cols = ['UDI', 'Product ID', 'Air temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Failure_Probability', 'Status']
        
        # Format probability as percentage for display
        display_df = results_df[display_cols].copy()
        display_df['Failure_Probability'] = (display_df['Failure_Probability'] * 100).round(1).astype(str) + '%'
        
        # Style the dataframe
        def color_status(val):
            color = '#d62728' if val == 'CRITICAL' else '#ff7f0e' if val == 'WARNING' else '#2ca02c'
            return f'color: {color}; font-weight: bold'
            
        st.dataframe(display_df.style.applymap(color_status, subset=['Status']), use_container_width=True)

    except Exception as e:
        st.error(f"Error loading fleet data: {e}")

# --- PAGE 2: WHAT-IF SIMULATION ---
elif page == "🎛️ What-If Simulation":
    st.title("🎛️ Interactive 'What-If' Simulation")
    st.markdown("Adjust operational sensor values to instantly simulate failure probability under heavy load or harsh weather conditions.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚙️ Sensor Inputs")
        air_temp = st.slider("Ambient Air Temp [K]", 290.0, 310.0, 298.1)
        proc_temp = st.slider("Process Temp [K]", 300.0, 320.0, 308.6)
        rot_speed = st.slider("Rotational Speed [rpm]", 1100, 3000, 1551)
        torque = st.slider("Torque [Nm]", 10.0, 80.0, 42.8)
        tool_wear = st.slider("Tool Wear [min]", 0, 300, 100)
        mach_type = st.selectbox("Machine Quality Type", options=[("Low", 1), ("Medium", 2), ("High", 0)], format_func=lambda x: x[0])
        
    with col2:
        st.subheader("🔮 Predictive Outcome")
        
        input_data = {
            "Air temperature [K]": air_temp,
            "Process temperature [K]": proc_temp,
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Type": mach_type[1]
        }
        
        # Run inference
        result = inference_engine.predict_risk(input_data)
        
        prob = result['failure_probability']
        status = result['status']
        
        if status == "CRITICAL":
            st.error("🚨 **CRITICAL RISK OF FAILURE**")
        elif status == "WARNING":
            st.warning("⚠️ **WARNING: Elevated Failure Risk**")
        else:
            st.success("✅ **STATUS NORMAL**")
            
        st.metric(label="Failure Probability", value=f"{prob:.1%}")
        st.progress(float(prob))
        
        # Temp Delta Alert
        temp_delta = proc_temp - air_temp
        st.info(f"💡 **Heat Dissipation:** Process is {temp_delta:.1f}K hotter than ambient.")
        if temp_delta > 12:
            st.warning("High temperature delta detected. HDF (Heat Dissipation Failure) risk elevated.")

# --- PAGE 3: SHAP EXPLAINABILITY ---
elif page == "🧠 SHAP Explainability":
    st.title("🧠 SHAP Root-Cause Analysis")
    st.markdown("Understand *why* the AI predicted a failure. This view allows **Reliability Engineers** to pinpoint exact root causes (e.g. Overstrain vs Heat Dissipation).")
    
    st.info("Simulating a High-Risk Machine for Explainability Analysis...")
    
    # High risk dummy data
    high_risk_data = {
        "Air temperature [K]": 302.3,
        "Process temperature [K]": 315.2, # high temp
        "Rotational speed [rpm]": 1300,
        "Torque [Nm]": 65.0, # high torque
        "Tool wear [min]": 220, # high wear
        "Type": 1
    }
    
    result = inference_engine.predict_risk(high_risk_data)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Machine State")
        st.json(high_risk_data)
        st.metric(label="Predicted Failure Risk", value=f"{result['failure_probability']:.1%}")
        
    with col2:
        st.subheader("Feature Importance (Local)")
        
        # Create Waterfall-like Bar chart for local SHAP using Plotly
        shap_vals = result['shap_values']
        features = list(shap_vals.keys())
        values = list(shap_vals.values())
        
        # Sort by absolute magnitude
        sorted_indices = sorted(range(len(values)), key=lambda i: abs(values[i]))
        sorted_features = [features[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        colors = ['red' if v > 0 else 'green' for v in sorted_values]
        
        fig = go.Figure(go.Bar(
            x=sorted_values,
            y=sorted_features,
            orientation='h',
            marker_color=colors
        ))
        
        fig.update_layout(
            title="SHAP Value impact on Failure Probability",
            xaxis_title="SHAP Value (Impact)",
            yaxis_title="Feature",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Red bars** drive the probability *up* (towards failure). **Green bars** drive the probability *down* (towards normal).")

# --- PAGE 4: THRESHOLD & NOISE ---
elif page == "⚙️ Threshold & Noise Tuning":
    st.title("⚙️ Week 4: Noise Sensitivity & Threshold Tuning")
    st.markdown("Mimic real-world deployment challenges by injecting noise into sensor data and adjusting decision boundaries to balance False Alarms vs Missed Failures.")
    
    threshold = st.slider("Decision Threshold (Probability)", 0.0, 1.0, 0.50, 0.05)
    noise_level = st.slider("Synthetic Sensor Noise (Std Dev)", 0.0, 10.0, 0.0, 1.0)
    
    st.info(f"Model will flag a failure if Probability > {threshold*100:.0f}%")
    
    # Just a visual placeholder demonstrating the concept for Week 4
    st.success(f"Configured threshold to {threshold}. In production, this shifts the precision-recall balance.")
    if noise_level > 0:
        st.warning(f"Injecting N(0, {noise_level}) noise into incoming telemetry. Model confidence may degrade.")
