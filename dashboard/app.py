"""
Contextual Predictive Maintenance Dashboard
Single-page, unified Streamlit application.
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="PredictMaint | Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# GLOBAL STYLES
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --gold: #d4af37;
        --gold-soft: rgba(212,175,55,0.35);
        --panel-bg: #101a30;
        --panel-bg-alt: #0d1729;
        --border: #223154;
        --border-soft: #1a2744;
        --text-main: #f2f5f9;
        --text-muted: #8fa0c2;
    }

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    .main {
        background:
            radial-gradient(1200px 500px at 10% -10%, #0f1a33 0%, transparent 60%),
            radial-gradient(1000px 500px at 90% 0%, #101c36 0%, transparent 55%),
            #0a1120;
    }
    .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1520px; }

    /* ---- Top header bar ---- */
    .header-wrap {
        display: flex; justify-content: space-between; align-items: center;
        background: linear-gradient(135deg, #101c38 0%, #0d1730 60%, #0f1c3a 100%);
        border: 1px solid var(--border); border-top: 1px solid var(--gold-soft);
        border-radius: 16px;
        padding: 20px 30px; margin-bottom: 24px;
        box-shadow: 0 10px 30px -12px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .header-title {
        font-family: 'Playfair Display', serif; font-size: 29px; font-weight: 700;
        color: var(--text-main); margin: 0; letter-spacing: 0.3px;
    }
    .header-sub {
        font-size: 12.5px; color: var(--gold); margin-top: 5px; font-weight: 600;
        letter-spacing: 1.4px; text-transform: uppercase;
    }
    .live-badge {
        background: rgba(34,197,94,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.4);
        padding: 7px 16px; border-radius: 20px; font-size: 12px; font-weight: 700;
        letter-spacing: 0.5px; box-shadow: 0 0 14px rgba(52,211,153,0.15);
    }
    .pill {
        background: #16223f; color: #cbd5e1; padding: 7px 15px; border-radius: 9px;
        font-size: 12.5px; margin-right: 8px; border: 1px solid var(--border);
        font-weight: 500;
    }

    /* ---- KPI cards ---- */
    .kpi-card {
        background: linear-gradient(160deg, var(--panel-bg) 0%, var(--panel-bg-alt) 100%);
        border: 1px solid var(--border); border-left: 3px solid var(--accent, var(--gold));
        border-radius: 12px;
        padding: 18px 20px; height: 144px; display: flex; flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 8px 20px -10px rgba(0,0,0,0.5);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 12px 26px -10px rgba(0,0,0,0.6); }
    .kpi-icon {
        width: 36px; height: 36px; border-radius: 9px; display: flex;
        align-items: center; justify-content: center; font-size: 17px; margin-bottom: 4px;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.05);
    }
    .kpi-label {
        font-size: 12.5px; color: var(--text-muted); font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.6px;
    }
    .kpi-value {
        font-family: 'Playfair Display', serif; font-size: 27px; font-weight: 700;
        color: var(--text-main); margin-top: 2px;
    }
    .kpi-delta-up { color: #f87171; font-size: 12px; font-weight: 600; }
    .kpi-delta-down { color: #34d399; font-size: 12px; font-weight: 600; }
    .kpi-delta-neutral { color: var(--text-muted); font-size: 12px; font-weight: 600; }

    /* ---- Section panels ---- */
    .panel {
        background: linear-gradient(160deg, var(--panel-bg) 0%, var(--panel-bg-alt) 100%);
        border: 1px solid var(--border); border-radius: 14px;
        padding: 20px 22px; margin-bottom: 22px;
        box-shadow: 0 10px 26px -14px rgba(0,0,0,0.55);
    }
    .panel-title {
        font-family: 'Playfair Display', serif; font-size: 17px; font-weight: 700;
        color: var(--text-main); margin-bottom: 14px; padding-bottom: 10px;
        border-bottom: 1px solid var(--border-soft);
        position: relative;
    }
    .panel-title::after {
        content: ""; position: absolute; left: 0; bottom: -1px; width: 46px; height: 2px;
        background: var(--gold);
    }
    .panel-link { color: #7db4ff; font-size: 12px; float: right; font-weight: 600; border-bottom: none; }

    /* ---- Status badges ---- */
    .badge-critical { background:rgba(127,29,29,0.55); color:#fca5a5; padding:4px 12px; border-radius:20px; font-size:11.5px; font-weight:700; border:1px solid #7f1d1d99; letter-spacing:0.3px;}
    .badge-warning  { background:rgba(120,53,15,0.55); color:#fcd34d; padding:4px 12px; border-radius:20px; font-size:11.5px; font-weight:700; border:1px solid #78350f99; letter-spacing:0.3px;}
    .badge-normal   { background:rgba(20,83,45,0.55); color:#86efac; padding:4px 12px; border-radius:20px; font-size:11.5px; font-weight:700; border:1px solid #14532d99; letter-spacing:0.3px;}

    /* ---- Data table ---- */
    .classic-table { width:100%; border-collapse:collapse; font-size:13px; }
    .classic-table thead tr { border-bottom: 2px solid var(--gold-soft); }
    .classic-table th {
        padding:9px 8px; text-align:left; color: var(--gold); font-weight:700;
        font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.6px;
    }
    .classic-table tbody tr { border-bottom: 1px solid var(--border-soft); transition: background 0.12s ease; }
    .classic-table tbody tr:nth-child(odd) { background: rgba(255,255,255,0.015); }
    .classic-table tbody tr:hover { background: rgba(212,175,55,0.06); }
    .classic-table td { padding:11px 8px; }

    /* ---- Alerts ---- */
    .alert-row {
        display:flex; align-items:flex-start; gap:12px; padding:12px 10px;
        border-left: 3px solid var(--border-soft); border-radius: 6px;
        margin-bottom: 8px; background: rgba(255,255,255,0.015);
        transition: background 0.12s ease;
    }
    .alert-row:hover { background: rgba(212,175,55,0.06); }
    .alert-row.critical { border-left-color: #ef4444; }
    .alert-row.warning { border-left-color: #f59e0b; }
    .alert-title { font-size: 13.5px; font-weight: 700; color: var(--text-main); }
    .alert-sub { font-size: 12px; color: var(--text-muted); margin-top: 2px; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1428 0%, #0a1120 100%);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] h3 {
        font-family: 'Playfair Display', serif; color: var(--gold) !important;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] { background: transparent; }
    div[data-testid="stMetric"] label { color: var(--text-muted) !important; }
    hr { border-color: var(--border-soft); }

    /* buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1c2b52 0%, #16223f 100%) !important;
        border: 1px solid var(--gold-soft) !important; color: var(--text-main) !important;
        font-weight: 600 !important; border-radius: 9px !important;
    }
    .stButton>button:hover { border-color: var(--gold) !important; color: var(--gold) !important; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# SYNTHETIC DATA LAYER  (swap this out for real data / model outputs)
# ----------------------------------------------------------------------------
@st.cache_data
def generate_machine_data(seed: int = 42, n_machines: int = 16, n_days: int = 7):
    rng = np.random.default_rng(seed)
    machine_ids = [f"M_{i:02d}" for i in range(1, n_machines + 1)]

    base_risk = np.clip(rng.beta(2, 5, n_machines), 0.02, 0.99)
    # force a couple of known high risk / low risk machines for a realistic story
    base_risk[6] = 0.93   # M_07
    base_risk[2] = 0.87   # M_03
    base_risk[0] = 0.76   # M_01
    base_risk[11] = 0.61  # M_12
    base_risk[4] = 0.42   # M_05

    machines = pd.DataFrame({
        "machine_id": machine_ids,
        "failure_probability": base_risk.round(2),
        "remaining_useful_life_days": np.round(np.clip(30 * (1 - base_risk) + rng.normal(0, 1.5, n_machines), 0.5, None), 1),
        "temperature_c": np.round(rng.normal(60, 6, n_machines) + base_risk * 15, 1),
        "vibration_mms": np.round(rng.normal(3.5, 0.8, n_machines) + base_risk * 3, 2),
        "load_pct": np.round(np.clip(rng.normal(65, 8, n_machines) + base_risk * 10, 20, 100), 1),
        "pressure_kpa": np.round(rng.normal(98, 6, n_machines), 1),
        "humidity_pct": np.round(rng.normal(46, 5, n_machines), 1),
        "last_update": [datetime(2024, 5, 7, 10, 0, 0)] * n_machines,
    })

    def status(p):
        if p >= 0.80:
            return "Critical"
        elif p >= 0.50:
            return "Warning"
        return "Normal"

    machines["status"] = machines["failure_probability"].apply(status)
    machines = machines.sort_values("failure_probability", ascending=False).reset_index(drop=True)

    dates = pd.date_range(end=datetime(2024, 5, 7), periods=n_days, freq="D")
    return machines, dates


@st.cache_data
def generate_horizon_series(machine_id: str, final_prob: float, horizon_label: str, seed: int = 99):
    """Builds a failure-probability trend sized to the selected prediction horizon."""
    # Deterministic per-machine offset (Python's built-in hash() is randomized
    # per process, which would make results shift on every app restart).
    machine_offset = sum(ord(c) for c in machine_id) % 1000
    rng = np.random.default_rng(seed + machine_offset)
    horizon_map = {
        "Next 24 Hours": ("h", 24),
        "Next 7 Days": ("D", 7),
        "Next 30 Days": ("D", 30),
    }
    freq, periods = horizon_map.get(horizon_label, ("D", 7))
    points = periods * 3 if freq == "D" else periods

    start_p = max(0.02, final_prob - rng.uniform(0.30, 0.50))
    trend = np.linspace(start_p, final_prob, points)
    noise = rng.normal(0, 0.015, points)
    series_vals = np.clip(trend + noise, 0, 1.3)

    end = datetime(2024, 5, 7, 10, 0, 0)
    idx = pd.date_range(end=end, periods=points, freq=freq)
    return pd.Series(series_vals, index=idx)


@st.cache_data
def generate_feature_importance(seed: int = 7):
    rng = np.random.default_rng(seed)
    features = ["vibration_rms", "temp_mean", "vibration_kurtosis", "load_mean", "pressure_mean"]
    importance = sorted(rng.uniform(0.08, 0.25, len(features)), reverse=True)
    return pd.DataFrame({"feature": features, "importance": np.round(importance, 2)})


@st.cache_data
def generate_shap_points(seed: int = 11, n_points: int = 120):
    rng = np.random.default_rng(seed)
    features = ["vibration_rms", "temp_mean", "vibration_kurtosis", "load_mean", "pressure_mean"]
    rows = []
    for i, f in enumerate(features):
        spread = 1.0 - i * 0.15
        vals = rng.normal(0, spread * 0.4, n_points)
        feat_val = rng.uniform(0, 1, n_points)  # used for color (low->high)
        for v, fv in zip(vals, feat_val):
            rows.append({"feature": f, "shap_value": v, "feature_value": fv})
    return pd.DataFrame(rows)


machines_df, date_range = generate_machine_data()
feat_imp_df = generate_feature_importance()
shap_df = generate_shap_points()

ALERT_THRESHOLD_DEFAULT = 50

# ----------------------------------------------------------------------------
# SIDEBAR — FILTERS  (wrapped in a form so "Apply Filters" has real effect;
# nothing below reruns until the button is pressed, matching classic BI tools)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ PREDICTMAINT")
    st.caption("IoT Edge AI · LightGBM Model")
    st.markdown("---")

    st.markdown("#### 🔍 Filters")
    with st.form("filters_form"):
        f_machine = st.selectbox("Select Machine", machines_df["machine_id"].tolist(), index=0)
        f_date_range = st.date_input(
            "Date Range",
            value=(date_range[0].date(), date_range[-1].date()),
            min_value=date_range[0].date(),
        )
        f_horizon = st.selectbox("Prediction Horizon", ["Next 24 Hours", "Next 7 Days", "Next 30 Days"], index=1)
        f_alert_threshold = st.slider("Alert Threshold (%)", 0, 100, ALERT_THRESHOLD_DEFAULT)
        submitted = st.form_submit_button("Apply Filters", use_container_width=True, type="primary")

    # Persist the last applied selection across reruns; seed sensible defaults
    # on first load so the dashboard isn't empty before the user clicks Apply.
    if submitted or "applied" not in st.session_state:
        st.session_state.applied = {
            "machine": f_machine,
            "date_range": f_date_range,
            "horizon": f_horizon,
            "alert_threshold": f_alert_threshold,
        }
        if submitted:
            st.toast("Filters applied", icon="✅")

    applied = st.session_state.applied
    selected_machine = applied["machine"]
    horizon = applied["horizon"]
    alert_threshold = applied["alert_threshold"]

    # date_input returns a single date while the user is mid-selection;
    # guard against that instead of crashing on a 2-tuple unpack.
    raw_range = applied["date_range"]
    if isinstance(raw_range, tuple) and len(raw_range) == 2:
        selected_start, selected_end = raw_range
    else:
        selected_start, selected_end = date_range[0].date(), date_range[-1].date()

    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.caption(
        "This dashboard predicts machine failure probability using IoT sensor "
        "data and contextual external data."
    )

    st.markdown("#### 📊 Model Info")
    st.caption("Model: **LightGBM**")
    st.caption("F1 Score: **0.9597**")
    st.caption("ROC AUC: **0.9923**")
    st.caption("PR AUC: **0.9876**")
    st.caption("Trained On: **2024-05-07**")
    st.success("● Model Status: Deployed")

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
n_at_risk = int((machines_df["failure_probability"] >= 0.80).sum())
n_alerts_7d = int((machines_df["failure_probability"] >= alert_threshold / 100).sum())
avg_fail_prob = machines_df["failure_probability"].mean() * 100
overall_health_pct = 100 - avg_fail_prob

st.markdown(f"""
<div class="header-wrap">
    <div>
        <p class="header-title">⚙️ Contextual Predictive Maintenance Dashboard</p>
        <p class="header-sub">IoT Edge AI &nbsp;·&nbsp; LightGBM Model &nbsp;·&nbsp; Real-time Monitoring</p>
    </div>
    <div style="display:flex; align-items:center; gap:14px;">
        <span class="pill">📅 {selected_end.strftime('%b %d, %Y')} &nbsp; 10:00:00 AM</span>
        <span class="pill">⏱️ {horizon}</span>
        <span class="pill">🔔 {n_alerts_7d} Alerts</span>
        <span class="live-badge">● LIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# KPI ROW
# ----------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)

health_label = "Fair" if overall_health_pct >= 70 else ("Good" if overall_health_pct >= 50 else "Poor")
health_color = "#22c55e" if overall_health_pct >= 70 else ("#f59e0b" if overall_health_pct >= 50 else "#ef4444")

with k1:
    st.markdown(f"""
    <div class="kpi-card" style="--accent:#22c55e;">
        <div class="kpi-icon" style="background:rgba(34,197,94,0.14); color:#34d399;">💚</div>
        <div class="kpi-label">Overall Health</div>
        <div class="kpi-value" style="color:{health_color};">{health_label}</div>
        <div class="kpi-delta-down">Risk: {avg_fail_prob:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card" style="--accent:#ef4444;">
        <div class="kpi-icon" style="background:rgba(239,68,68,0.14); color:#f87171;">🛡️</div>
        <div class="kpi-label">Machines at Risk</div>
        <div class="kpi-value" style="color:#f87171;">{n_at_risk}</div>
        <div class="kpi-delta-up">&gt;80% Failure Prob.</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card" style="--accent:#3b82f6;">
        <div class="kpi-icon" style="background:rgba(59,130,246,0.14); color:#60a5fa;">🖥️</div>
        <div class="kpi-label">Total Machines</div>
        <div class="kpi-value">{len(machines_df)}</div>
        <div class="kpi-delta-neutral">Active</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card" style="--accent:#f59e0b;">
        <div class="kpi-icon" style="background:rgba(245,158,11,0.14); color:#fb923c;">🔔</div>
        <div class="kpi-label">Alerts (Next 7 Days)</div>
        <div class="kpi-value" style="color:#fb923c;">{n_alerts_7d}</div>
        <div class="kpi-delta-up">High Priority</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="kpi-card" style="--accent:#a855f7;">
        <div class="kpi-icon" style="background:rgba(168,85,247,0.14); color:#c084fc;">📈</div>
        <div class="kpi-label">Avg. Failure Probability</div>
        <div class="kpi-value" style="color:#c084fc;">{avg_fail_prob:.1f}%</div>
        <div class="kpi-delta-up">Across All Machines</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------------------------------
# ROW 2 — Failure prob over time | Distribution donut | Top features
# ----------------------------------------------------------------------------
c1, c2, c3 = st.columns([1.3, 1, 1.1])

with c1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f'<p class="panel-title">Failure Probability Over Time ({selected_machine} · {horizon})</p>', unsafe_allow_html=True)

    final_prob = float(machines_df.loc[machines_df["machine_id"] == selected_machine, "failure_probability"].iloc[0])
    series = generate_horizon_series(selected_machine, final_prob, horizon)

    # Clip to the applied date range for daily-resolution horizons; hourly
    # ("Next 24 Hours") view ignores the date picker since it's sub-daily.
    if horizon != "Next 24 Hours":
        mask = (series.index.date >= selected_start) & (series.index.date <= selected_end)
        if mask.any():
            series = series[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines+markers",
        line=dict(color="#3b82f6", width=2.5), marker=dict(size=5, color="#3b82f6"),
        name="Failure Probability", fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"
    ))
    fig.add_hline(y=alert_threshold / 100, line_dash="dash", line_color="#ef4444",
                   annotation_text=f"Threshold ({alert_threshold/100:.2f})",
                   annotation_font_color="#ef4444", annotation_position="top left")
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        xaxis=dict(showgrid=False, tickformat="%H:%M" if horizon == "Next 24 Hours" else "%b %d"),
        yaxis=dict(title="Failure Probability", range=[0, 1.05], gridcolor="#1f2b45"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Failure Probability Distribution</p>', unsafe_allow_html=True)
    bins = pd.cut(machines_df["failure_probability"], bins=[-0.01, 0.2, 0.5, 0.8, 1.01],
                  labels=["0-20%", "20-50%", "50-80%", "80-100%"])
    dist = bins.value_counts().reindex(["0-20%", "20-50%", "50-80%", "80-100%"]).fillna(0)
    colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]
    fig2 = go.Figure(data=[go.Pie(
        labels=dist.index, values=dist.values, hole=0.62,
        marker=dict(colors=colors, line=dict(color="#111a2e", width=2)),
        textinfo="percent", textfont=dict(color="white", size=12),
    )])
    fig2.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
        showlegend=True, legend=dict(orientation="v", font=dict(size=11)),
        annotations=[dict(text=f"{len(machines_df)}<br>Machines", x=0.5, y=0.5,
                           font_size=16, font_color="#f1f5f9", showarrow=False)],
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Top 5 Contributing Features</p>', unsafe_allow_html=True)
    fi = feat_imp_df.sort_values("importance")
    fig3 = go.Figure(go.Bar(
        x=fi["importance"], y=fi["feature"], orientation="h",
        marker=dict(color=fi["importance"], colorscale=["#22c55e", "#3b82f6", "#a855f7"]),
        text=fi["importance"], textposition="outside", textfont=dict(color="#cbd5e1"),
    ))
    fig3.update_layout(
        height=300, margin=dict(l=10, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        xaxis=dict(title="Feature Importance (Gain)", gridcolor="#1f2b45"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# ROW 3 — Top machines at risk table | Recent alerts
# ----------------------------------------------------------------------------
t1, t2 = st.columns([1.6, 1])

with t1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Top Machines at Risk</p>', unsafe_allow_html=True)

    top_risk = machines_df.head(5).copy()

    def badge(status):
        cls = {"Critical": "badge-critical", "Warning": "badge-warning", "Normal": "badge-normal"}[status]
        return f'<span class="{cls}">{status}</span>'

    rows_html = ""
    for _, r in top_risk.iterrows():
        prob_color = "#f87171" if r["failure_probability"] >= 0.8 else ("#fbbf24" if r["failure_probability"] >= 0.5 else "#4ade80")
        rows_html += f"""
        <tr>
            <td style="font-weight:700; color:#f2f5f9;">{r['machine_id']}</td>
            <td style="color:{prob_color}; font-weight:700;">{r['failure_probability']:.2f}</td>
            <td style="color:#cbd5e1;">{r['remaining_useful_life_days']} days</td>
            <td style="color:#cbd5e1;">{r['temperature_c']}</td>
            <td style="color:#cbd5e1;">{r['vibration_mms']}</td>
            <td style="color:#cbd5e1;">{r['load_pct']}</td>
            <td style="color:#8fa0c2; font-size:12px;">{r['last_update'].strftime('%Y-%m-%d %H:%M:%S')}</td>
            <td>{badge(r['status'])}</td>
        </tr>"""

    st.markdown(f"""
    <table class="classic-table">
        <thead>
            <tr>
                <th>Machine ID</th>
                <th>Failure Prob.</th>
                <th>RUL</th>
                <th>Temp (°C)</th>
                <th>Vibration</th>
                <th>Load (%)</th>
                <th>Last Update</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with t2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Recent Alerts <span class="panel-link">View All</span></p>', unsafe_allow_html=True)

    alert_icons = {"Critical": "🔴", "Warning": "🟠"}
    alert_css_class = {"Critical": "critical", "Warning": "warning"}
    alerts_html = ""
    for _, r in machines_df[machines_df["status"] != "Normal"].head(3).iterrows():
        icon = alert_icons.get(r["status"], "🟡")
        cls = alert_css_class.get(r["status"], "")
        alerts_html += f"""
        <div class="alert-row {cls}">
            <div>{icon}</div>
            <div>
                <div class="alert-title">High failure probability for {r['machine_id']}</div>
                <div class="alert-sub">Probability: {r['failure_probability']:.2f} &nbsp;|&nbsp; Time: {r['last_update'].strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
        </div>"""
    st.markdown(alerts_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# ROW 4 — Machine detail sensors | Gauge | SHAP summary
# ----------------------------------------------------------------------------
d1, d2, d3 = st.columns([1, 0.9, 1.3])

sel_row = machines_df[machines_df["machine_id"] == selected_machine].iloc[0]

with d1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f'<p class="panel-title">Machine Details ({selected_machine})</p>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("🌡️ Temperature", f"{sel_row['temperature_c']} °C")
    m2.metric("📳 Vibration (RMS)", f"{sel_row['vibration_mms']} mm/s")
    m3, m4 = st.columns(2)
    m3.metric("⚡ Load", f"{sel_row['load_pct']} %")
    m4.metric("🎚️ Pressure", f"{sel_row['pressure_kpa']} kPa")
    st.metric("💧 Humidity", f"{sel_row['humidity_pct']} %")
    st.markdown('</div>', unsafe_allow_html=True)

with d2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Predicted Failure Probability (Next 7 Days)</p>', unsafe_allow_html=True)
    prob_pct = sel_row["failure_probability"] * 100
    risk_level = "High" if prob_pct >= 70 else ("Medium" if prob_pct >= 40 else "Low")
    gauge_color = "#ef4444" if prob_pct >= 70 else ("#f59e0b" if prob_pct >= 40 else "#22c55e")
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={"suffix": "%", "font": {"size": 34, "color": "#f1f5f9"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b98b5"},
            "bar": {"color": gauge_color},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 40], "color": "#14532d"},
                {"range": [40, 70], "color": "#78350f"},
                {"range": [70, 100], "color": "#7f1d1d"},
            ],
        },
    ))
    fig_g.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=0),
                         paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1")
    st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})
    st.markdown(f'<p style="text-align:center; font-weight:700; color:{gauge_color};">Risk Level: {risk_level}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with d3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">SHAP Summary (Feature Impact)</p>', unsafe_allow_html=True)

    # Built manually with go.Scatter (jittered beeswarm) instead of px.strip,
    # since px.strip's color_continuous_scale argument is not supported on
    # all installed Plotly versions and raised a TypeError.
    ordered_features = feat_imp_df.sort_values("importance")["feature"].tolist()
    rng_jitter = np.random.default_rng(3)
    fig4 = go.Figure()
    for i, feat in enumerate(ordered_features):
        sub = shap_df[shap_df["feature"] == feat]
        jitter = rng_jitter.uniform(-0.32, 0.32, len(sub))
        fig4.add_trace(go.Scatter(
            x=sub["shap_value"],
            y=[i + j for j in jitter],
            mode="markers",
            marker=dict(
                size=6, opacity=0.75,
                color=sub["feature_value"],
                colorscale=["#3b82f6", "#a855f7", "#ef4444"],
                cmin=0, cmax=1,
                showscale=(i == 0),
                colorbar=dict(title="Value", tickvals=[0, 1], ticktext=["Low", "High"]) if i == 0 else None,
            ),
            name=feat,
            showlegend=False,
            hovertemplate=f"{feat}<br>SHAP: %{{x:.3f}}<extra></extra>",
        ))
    fig4.add_vline(x=0, line_color="#3a4568", line_width=1)
    fig4.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        xaxis=dict(title="SHAP value (impact on model output)", gridcolor="#1f2b45", zerolinecolor="#3a4568"),
        yaxis=dict(
            title="", tickmode="array",
            tickvals=list(range(len(ordered_features))),
            ticktext=ordered_features,
            range=[-0.6, len(ordered_features) - 0.4],
            gridcolor="#1f2b45",
        ),
    )
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown(
    "<p style='text-align:center; color:#4b5875; font-size:12px; margin-top:10px;'>"
    "© 2024 PredictMaint. All rights reserved. &nbsp;·&nbsp; Version 2.0.0"
    "</p>", unsafe_allow_html=True
)

