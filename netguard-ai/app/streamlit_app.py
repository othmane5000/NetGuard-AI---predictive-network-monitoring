"""
NETGUARD AI - Network Operations Center Dashboard
====================================================
Run with:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

from src.prediction.predict import NetGuardPredictor
from src.evaluation.metrics import get_confusion_matrix, get_roc_curve, classification_report_manual

st.set_page_config(
    page_title="NetGuard AI | NOC Dashboard",
    page_icon="\U0001F6F0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Modern Premium SaaS Visual Theme
# ----------------------------------------------------------------------------
RISK_COLORS = {
    "LOW": "#22c55e",
    "MEDIUM": "#eab308",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    /* Global Theme & Background */
    .stApp { 
        background-color: #0b0f19 !important; 
        color: #f1f5f9;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    /* Ambient Ambient Glow */
    .stApp::before {
        content: "";
        position: fixed;
        top: -100px;
        left: 30%;
        width: 600px;
        height: 400px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.07) 0%, rgba(0,0,0,0) 70%);
        pointer-events: none;
        z-index: 0;
    }

    /* Modern Sidebar */
    [data-testid="stSidebar"] { 
        background-color: #111827 !important;
        border-right: 1px solid #1f2937 !important;
    }

    /* Modern Typography */
    h1, h2, h3, h4 { 
        font-family: 'Plus Jakarta Sans', sans-serif !important; 
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #f8fafc !important;
    }

    .noc-header {
        font-family: 'JetBrains Mono', monospace; 
        font-size: 11px; 
        color: #818cf8;
        text-transform: uppercase; 
        letter-spacing: 2px; 
        margin-bottom: 2px;
        font-weight: 700;
    }

    /* Clean KPI Cards */
    .kpi-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #374151;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
    }
    .kpi-value { 
        font-family: 'JetBrains Mono', monospace; 
        font-size: 30px; 
        font-weight: 700; 
        line-height: 1.2;
    }
    .kpi-label { 
        color: #9ca3af; 
        font-size: 11px; 
        font-weight: 600;
        text-transform: uppercase; 
        letter-spacing: 1px; 
        margin-top: 6px;
    }

    /* Clean Status Badges */
    .risk-pill {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px; 
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace; 
        font-size: 11px; 
        font-weight: 700;
    }

    /* Streamlit UI Controls Overrides */
    div[data-testid="stMetricValue"] { 
        font-family: 'JetBrains Mono', monospace !important; 
        font-weight: 700 !important;
    }
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 12px 16px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding: 0;
        border-bottom: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 6px 6px 0 0;
        color: #94a3b8;
        font-weight: 600;
        font-size: 13px;
        border: none !important;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border-bottom: 2px solid #6366f1 !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #1f2937;
        border-radius: 10px;
        overflow: hidden;
    }
    hr {
        border-color: #1f2937 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0b0f19; }
    ::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #374151; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return NetGuardPredictor()


@st.cache_data
def load_raw_data():
    return pd.read_csv(BASE / "data" / "raw" / "network_telemetry.csv", parse_dates=["timestamp"])


@st.cache_data
def compute_assessment(_predictor, raw_df):
    return _predictor.assess_all_devices(raw_df)


def risk_pill(risk_level):
    color = RISK_COLORS.get(risk_level, "#64748b")
    return f'<span class="risk-pill" style="background:{color}1e;color:{color};border:1px solid {color}40;">{risk_level}</span>'


predictor = load_predictor()
raw_df = load_raw_data()
assessment = compute_assessment(predictor, raw_df)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.markdown('<div class="noc-header">NetGuard AI</div>', unsafe_allow_html=True)
st.sidebar.title("\U0001F6F0 Network Operations Center")
page = st.sidebar.radio("Navigate", ["Dashboard", "Device Monitoring", "Device Details", "Analytics"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Risk thresholds** (configurable)")
st.sidebar.caption("LOW 0-30% · MEDIUM 30-60% · HIGH 60-80% · CRITICAL 80-100%")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Predictions reflect metric-based risk indicators only. NetGuard AI "
    "does not identify specific failing physical components."
)

# ----------------------------------------------------------------------------
# PAGE: Dashboard
# ----------------------------------------------------------------------------
if page == "Dashboard":
    st.markdown('<div class="noc-header">Fleet Overview</div>', unsafe_allow_html=True)
    st.title("Network Health Overview")

    total = len(assessment)
    healthy = (assessment["risk_level"] == "LOW").sum()
    warning = (assessment["risk_level"] == "MEDIUM").sum()
    high = (assessment["risk_level"] == "HIGH").sum()
    critical = (assessment["risk_level"] == "CRITICAL").sum()
    predicted_failures = (assessment["failure_probability"] >= 0.5).sum()

    cols = st.columns(6)
    kpi_data = [
        ("Total Devices", total, "#e2e8f0"),
        ("Healthy", healthy, RISK_COLORS["LOW"]),
        ("Warning", warning, RISK_COLORS["MEDIUM"]),
        ("High Risk", high, RISK_COLORS["HIGH"]),
        ("Critical", critical, RISK_COLORS["CRITICAL"]),
        ("Predicted Failures", predicted_failures, "#a855f7"),
    ]
    for col, (label, value, color) in zip(cols, kpi_data):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:{color}">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Highest-risk devices")
        top = assessment.sort_values("failure_probability", ascending=False).head(10)
        for _, row in top.iterrows():
            with st.container():
                a, b, c, d = st.columns([2, 1.3, 1.3, 3])
                a.markdown(f"**{row['device_id']}**  \n`{row['device_type']}`")
                b.markdown(f"Prob: **{row['failure_probability']:.0%}**")
                c.markdown(risk_pill(row["risk_level"]), unsafe_allow_html=True)
                indicators_str = ", ".join(i["label"] for i in row["indicators"][:3])
                d.markdown(f"<span style='color:#94a3b8'>{indicators_str}</span>", unsafe_allow_html=True)
                st.markdown("<hr style='margin:4px 0;border-color:#1f2937'>", unsafe_allow_html=True)

    with c2:
        st.subheader("Risk distribution")
        dist = assessment["risk_level"].value_counts().reindex(
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        colors = [RISK_COLORS[l] for l in dist.index]
        wedges, _ = ax.pie(dist.values, colors=colors, startangle=90,
                             wedgeprops=dict(width=0.4, edgecolor="#0b1220"))
        ax.legend(wedges, [f"{l} ({int(v)})" for l, v in dist.items()],
                   loc="center", fontsize=9, frameon=False, labelcolor="#e2e8f0")
        st.pyplot(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: Device Monitoring
# ----------------------------------------------------------------------------
elif page == "Device Monitoring":
    st.title("Device Monitoring")

    f1, f2, f3 = st.columns(3)
    type_filter = f1.multiselect("Device type", sorted(assessment["device_type"].unique()))
    risk_filter = f2.multiselect("Risk level", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    search = f3.text_input("Search device ID")

    view = assessment.copy()
    if type_filter:
        view = view[view["device_type"].isin(type_filter)]
    if risk_filter:
        view = view[view["risk_level"].isin(risk_filter)]
    if search:
        view = view[view["device_id"].str.contains(search, case=False)]

    view = view.sort_values("failure_probability", ascending=False)
    display = view[["device_id", "device_type", "cpu_usage", "memory_usage", "temperature",
                     "packet_loss", "latency", "failure_probability", "risk_level"]].copy()
    display["failure_probability"] = (display["failure_probability"] * 100).round(1).astype(str) + "%"
    display.columns = ["Device ID", "Type", "CPU %", "Mem %", "Temp °C", "Pkt Loss %",
                        "Latency ms", "Failure Prob.", "Risk"]

    st.dataframe(
        display.style.map(
            lambda v: f"color: {RISK_COLORS.get(v, '#e2e8f0')}; font-weight:700" if v in RISK_COLORS else "",
            subset=["Risk"],
        ),
        use_container_width=True, height=560,
    )

# ----------------------------------------------------------------------------
# PAGE: Device Details
# ----------------------------------------------------------------------------
elif page == "Device Details":
    st.title("Device Details")

    device_id = st.selectbox("Select device", sorted(assessment["device_id"].unique()))
    row = assessment[assessment["device_id"] == device_id].iloc[0]
    history = raw_df[raw_df["device_id"] == device_id].sort_values("timestamp")

    top_row = st.columns([1, 1, 1, 2])
    top_row[0].metric("Failure Probability", f"{row['failure_probability']:.1%}")
    top_row[1].markdown(f"**Risk Level**  \n{risk_pill(row['risk_level'])}", unsafe_allow_html=True)
    top_row[2].metric("Device Type", row["device_type"])
    top_row[3].markdown(f"**Prediction**  \n{row['prediction']}")

    st.markdown("### Current metrics")
    m = st.columns(5)
    m[0].metric("CPU Usage", f"{row['cpu_usage']:.1f}%")
    m[1].metric("Memory Usage", f"{row['memory_usage']:.1f}%")
    m[2].metric("Temperature", f"{row['temperature']:.1f}°C")
    m[3].metric("Packet Loss", f"{row['packet_loss']:.2f}%")
    m[4].metric("Latency", f"{row['latency']:.1f} ms")

    st.markdown("### Why is this device at risk?")
    for ind in row["indicators"]:
        color = {"VERY HIGH": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#22c55e"}.get(ind["severity"], "#94a3b8")
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1f2937'>"
            f"<span>{ind['label']}</span><span style='color:{color};font-weight:700'>{ind['severity']}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Recommended actions")
    for action in row["recommended_actions"]:
        st.markdown(f"- {action}")

    st.markdown("### Historical trends (last 30 days)")
    tabs = st.tabs(["CPU / Memory / Temp", "Packet Loss / Latency", "Interface Errors"])
    with tabs[0]:
        fig, ax = plt.subplots(figsize=(10, 3.2))
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        ax.plot(history["timestamp"], history["cpu_usage"], label="CPU %", color="#3b82f6")
        ax.plot(history["timestamp"], history["memory_usage"], label="Memory %", color="#a855f7")
        ax.plot(history["timestamp"], history["temperature"], label="Temp °C", color="#f97316")
        ax.legend(facecolor="#0b1220", labelcolor="#e2e8f0"); ax.tick_params(colors="#94a3b8")
        for s in ax.spines.values(): s.set_color("#1f2937")
        st.pyplot(fig, use_container_width=True)
    with tabs[1]:
        fig, ax = plt.subplots(figsize=(10, 3.2))
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        ax.plot(history["timestamp"], history["packet_loss"], label="Packet Loss %", color="#ef4444")
        ax2 = ax.twinx()
        ax2.plot(history["timestamp"], history["latency"], label="Latency ms", color="#22d3ee")
        ax.legend(loc="upper left", facecolor="#0b1220", labelcolor="#e2e8f0")
        ax2.legend(loc="upper right", facecolor="#0b1220", labelcolor="#e2e8f0")
        ax.tick_params(colors="#94a3b8"); ax2.tick_params(colors="#94a3b8")
        for s in ax.spines.values(): s.set_color("#1f2937")
        st.pyplot(fig, use_container_width=True)
    with tabs[2]:
        fig, ax = plt.subplots(figsize=(10, 3.2))
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        ax.bar(history["timestamp"], history["interface_errors"], color="#eab308", width=0.03)
        ax.tick_params(colors="#94a3b8")
        for s in ax.spines.values(): s.set_color("#1f2937")
        st.pyplot(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: Analytics
# ----------------------------------------------------------------------------
elif page == "Analytics":
    st.title("Model Analytics")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Failure & Risk Distribution", "Model Performance", "Feature Importance", "Historical Trends"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Failure type distribution (historical)")
            counts = raw_df[raw_df["failure"] == 1]["failure_type"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_alpha(0); ax.set_facecolor("none")
            ax.bar(counts.index, counts.values, color=["#ef4444", "#f97316", "#eab308"])
            ax.tick_params(colors="#94a3b8", rotation=20)
            for s in ax.spines.values(): s.set_color("#1f2937")
            st.pyplot(fig, use_container_width=True)
        with c2:
            st.subheader("Current fleet risk distribution")
            dist = assessment["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_alpha(0); ax.set_facecolor("none")
            ax.bar(dist.index, dist.values, color=[RISK_COLORS[l] for l in dist.index])
            ax.tick_params(colors="#94a3b8")
            for s in ax.spines.values(): s.set_color("#1f2937")
            st.pyplot(fig, use_container_width=True)

    with tab2:
        import json
        with open(BASE / "reports" / "classification_results.json") as f:
            results = json.load(f)
        st.subheader("Model comparison")
        comp_df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
        st.dataframe(comp_df.style.format("{:.3f}").background_gradient(cmap="Blues", axis=0),
                     use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Confusion Matrix (production model)")
            cm = np.load(BASE / "models" / "confusion_matrix.npy")
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_alpha(0)
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["No Failure", "Failure"], color="#e2e8f0")
            ax.set_yticklabels(["No Failure", "Failure"], color="#e2e8f0")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                             color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
            st.pyplot(fig, use_container_width=True)
        with c2:
            st.subheader("ROC Curve")
            data = np.load(BASE / "models" / "roc_curve.npz")
            fpr, tpr = data["fpr"], data["tpr"]
            auc = np.trapezoid(tpr, fpr)
            fig, ax = plt.subplots(figsize=(4.3, 4))
            fig.patch.set_alpha(0); ax.set_facecolor("none")
            ax.plot(fpr, tpr, color="#3b82f6", label=f"AUC={auc:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="#64748b")
            ax.legend(facecolor="#0b1220", labelcolor="#e2e8f0")
            ax.tick_params(colors="#94a3b8")
            for s in ax.spines.values(): s.set_color("#1f2937")
            st.pyplot(fig, use_container_width=True)

    with tab3:
        st.subheader("Top risk indicators (logistic regression coefficients)")
        imp = pd.read_csv(BASE / "models" / "feature_importance.csv", index_col=0).head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        colors = ["#ef4444" if v > 0 else "#3b82f6" for v in imp["coefficient"]]
        ax.barh(imp.index[::-1], imp["coefficient"][::-1], color=colors[::-1])
        ax.tick_params(colors="#94a3b8")
        for s in ax.spines.values(): s.set_color("#1f2937")
        st.pyplot(fig, use_container_width=True)
        st.caption("Red = increases failure risk · Blue = decreases failure risk (standardized coefficients)")

    with tab4:
        st.subheader("Fleet-wide average metrics over time")
        trend = raw_df.groupby("timestamp")[["cpu_usage", "temperature", "packet_loss"]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        ax.plot(trend["timestamp"], trend["cpu_usage"], label="Avg CPU %", color="#3b82f6")
        ax.plot(trend["timestamp"], trend["temperature"], label="Avg Temp °C", color="#f97316")
        ax.plot(trend["timestamp"], trend["packet_loss"] * 5, label="Avg Packet Loss % (x5)", color="#ef4444")
        ax.legend(facecolor="#0b1220", labelcolor="#e2e8f0")
        ax.tick_params(colors="#94a3b8")
        for s in ax.spines.values(): s.set_color("#1f2937")
        st.pyplot(fig, use_container_width=True)