# deadlock_lstm_app_smooth.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
import streamlit as st
from collections import deque
from tensorflow.keras.models import load_model
from deadlock_utils import generate_snapshot
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Smooth Temporal Deadlock Predictor", layout="wide")

# -----------------------------
# Static header (won't trigger scroll jump)
# -----------------------------
st.title("⚡ Smooth Temporal Deadlock Predictor (LSTM)")
st.sidebar.title("Simulation Configuration")

# -----------------------------
# Sidebar inputs
# -----------------------------
num_processes = st.sidebar.slider("Number of Processes", 2, 20, 10)
num_resources = st.sidebar.slider("Number of Resources", 1, 10, 5)
threshold = st.sidebar.slider("Deadlock Probability Threshold", 0.0, 1.0, 0.5)
update_interval = st.sidebar.slider("Update Interval (seconds)", 0.5, 5.0, 1.0)
sequence_length = st.sidebar.slider("Sequence Length for Prediction", 2, 10, 5)
max_snapshots = 20

# -----------------------------
# Load trained LSTM model
# -----------------------------
@st.cache_resource
def load_lstm(path="lstm_deadlock_model.h5"):
    return load_model(path)

lstm_model = load_lstm()

# -----------------------------
# Initialize session state
# -----------------------------
if "snapshot_deque" not in st.session_state:
    st.session_state.snapshot_deque = deque(maxlen=sequence_length)
if "probs" not in st.session_state:
    st.session_state.probs = deque(maxlen=max_snapshots)
if "high_priority_blocked_history" not in st.session_state:
    st.session_state.high_priority_blocked_history = deque(maxlen=max_snapshots)
if "deadlock_risk_history" not in st.session_state:
    st.session_state.deadlock_risk_history = deque(maxlen=max_snapshots)
if "steps" not in st.session_state:
    st.session_state.steps = 0

# -----------------------------
# Containers for dynamic updates
# -----------------------------
prob_container = st.empty()
dep_container = st.empty()
st.markdown("### 🟢 Current Allocation Matrix")
alloc_container = st.empty()
st.markdown("### 📈 Deadlock Probability Over Time")
chart_container = st.empty()
st.markdown("### 📊 Metrics & Stats")
metrics_container = st.empty()
st.markdown("### 📜 Deadlock Prediction History")
history_container = st.empty()

# -----------------------------
# Initialize Plotly chart
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    y=[],
    mode='lines+markers',
    line=dict(color='cyan', shape='spline'),
    name="Deadlock Probability"
))
fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold", annotation_position="top right")
fig.update_layout(
    xaxis_title="Step",
    yaxis_title="Deadlock Probability",
    yaxis=dict(range=[0, 1]),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color="white", size=10),
    showlegend=True
)
chart_container.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Real-time update loop
# -----------------------------
while True:
    st.session_state.steps += 1

    # Generate snapshot & predict
    features, allocated, dep_depth, deadlock_risk, high_priority_blocked = generate_snapshot(num_processes, num_resources)
    st.session_state.snapshot_deque.append(features)

    if len(st.session_state.snapshot_deque) == sequence_length:
        input_seq = np.array(st.session_state.snapshot_deque).reshape(1, sequence_length, len(features))
        prob = lstm_model.predict(input_seq, verbose=0)[0][0]
    else:
        prob = 0.0

    # Update rolling stats
    st.session_state.probs.append(prob)
    st.session_state.high_priority_blocked_history.append(high_priority_blocked)
    st.session_state.deadlock_risk_history.append(deadlock_risk)

    # -----------------------------
    # Update dashboard
    # -----------------------------
    if prob > threshold:
        prob_container.error(f"⚠️ High Deadlock Risk! Probability: {prob:.2f}")
    elif prob > threshold / 2:
        prob_container.warning(f"⚠️ Medium Risk. Probability: {prob:.2f}")
    else:
        prob_container.success(f"✅ System Safe. Probability: {prob:.2f}")

    dep_container.write(f"Dependency Depth: {dep_depth}")
    alloc_container.table(allocated)

    # -----------------------------
    # Update Plotly chart
    # -----------------------------
    fig.data[0].y = list(st.session_state.probs)
    fig.data[0].x = list(range(1, len(st.session_state.probs)+1))
    chart_container.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Update metrics
    # -----------------------------
    rolling_avg = np.mean(list(st.session_state.probs))
    rolling_max = np.max(list(st.session_state.probs))
    rolling_min = np.min(list(st.session_state.probs))
    metrics_container.markdown(f"""
    - **Rolling Average Probability:** {rolling_avg:.2f}  
    - **Max Probability:** {rolling_max:.2f}  
    - **Min Probability:** {rolling_min:.2f}  
    - **High-Priority Blocked Processes:** {high_priority_blocked}
    """)

    # -----------------------------
    # Update history table
    # -----------------------------
    history_data = {
        "Step": list(range(st.session_state.steps - len(st.session_state.deadlock_risk_history)+1, st.session_state.steps+1)),
        "Deadlock Risk": list(st.session_state.deadlock_risk_history),
        "Probability": list(st.session_state.probs),
        "High-Priority Blocked": list(st.session_state.high_priority_blocked_history)
    }
    history_container.dataframe(history_data, width=1000)

    # Wait before next update
    time.sleep(update_interval)
