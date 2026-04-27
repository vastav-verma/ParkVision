"""
🅿️ Parking Demand Prediction — Alumni Meet & Conferences
=========================================================
Streamlit Application | Full Featured Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parking Demand Predictor",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d1117; }

h1,h2,h3 { font-family: 'Syne', sans-serif !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1923 0%, #111827 100%);
    border-right: 1px solid #1e2d3d;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #131f2e 0%, #1a2840 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 800;
    color: #38bdf8; line-height: 1.1;
}
.metric-label {
    font-size: 0.78rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-top: 4px;
}
.metric-sub {
    font-size: 0.85rem; color: #94a3b8; margin-top: 6px;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 14px 32px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.05rem !important; font-weight: 700 !important;
    width: 100% !important; letter-spacing: 0.5px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(59,130,246,0.6) !important;
}

/* Alert boxes */
.alert-high   { background:#1f1010; border:1px solid #ef4444; border-radius:12px; padding:16px; color:#fca5a5; }
.alert-medium { background:#1a1a0d; border:1px solid #f59e0b; border-radius:12px; padding:16px; color:#fcd34d; }
.alert-low    { background:#0d1f14; border:1px solid #22c55e; border-radius:12px; padding:16px; color:#86efac; }

/* Section header */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 700;
    color: #e2e8f0; border-left: 4px solid #3b82f6;
    padding-left: 14px; margin: 28px 0 16px;
}

/* Prediction result banner */
.result-banner {
    background: linear-gradient(135deg, #0c1f3a, #0a2444);
    border: 2px solid #3b82f6;
    border-radius: 20px; padding: 32px;
    text-align: center; margin: 20px 0;
}
.result-number {
    font-family: 'Syne', sans-serif;
    font-size: 4rem; font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.result-label {
    font-size: 1rem; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 2px; margin-top: -6px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background: #0d1117 !important; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: #131f2e !important; color: #64748b !important;
    border-radius: 10px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9) !important;
    color: white !important;
}

/* Inputs */
.stSelectbox > div > div, .stNumberInput > div > div > input,
.stSlider > div { color: #e2e8f0 !important; }

div[data-testid="stExpander"] { background: #131f2e; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_path = "models/parking_model.pkl"
    if not os.path.exists(model_path):
        # Try relative to this file
        base = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base, "models", "parking_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    csv_path = "data/parking_dataset.csv"
    if not os.path.exists(csv_path):
        base = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base, "data", "parking_dataset.csv")
    return pd.read_csv(csv_path)


try:
    artifacts = load_artifacts()
    df = load_data()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"❌ Could not load model: {e}\n\nRun `python utils/model_trainer.py` first.")
    st.stop()

best_model      = artifacts["best_model"]
best_model_name = artifacts["best_model_name"]
all_models      = artifacts["all_models"]
encoders        = artifacts["encoders"]
scaler          = artifacts["scaler"]
feature_cols    = artifacts["feature_cols"]
results_df      = artifacts["results_df"]
fi_df           = artifacts["feature_importance"]

EVENT_TYPES   = sorted(df["event_type"].unique())
VENUES        = sorted(df["venue"].unique())
CITIES        = sorted(df["city"].unique())
WEATHERS      = sorted(df["weather"].unique())
REG_MODES     = sorted(df["registration_mode"].unique())


# ── Prediction helper ───────────────────────────────────────────────────────────
def predict(input_dict: dict, model_name: str = None) -> float:
    model = all_models[model_name] if model_name else best_model
    row = pd.DataFrame([input_dict])

    for col in artifacts["categorical_cols"]:
        if col in row.columns:
            le = encoders[col]
            val = str(row[col].iloc[0])
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                row[col] = 0

    X = row[feature_cols]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return max(0, round(pred))


def demand_category(demand, capacity):
    ratio = demand / capacity
    if ratio >= 0.9:    return "🔴 Critical", "alert-high",   "Overflow likely! Arrange off-site parking."
    elif ratio >= 0.7:  return "🟡 High",     "alert-medium", "Reserve additional spaces in advance."
    else:               return "🟢 Normal",    "alert-low",    "Parking supply is adequate."


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
        <div style='font-size:3rem;'>🅿️</div>
        <div style='font-family:Syne; font-size:1.2rem; font-weight:800; color:#e2e8f0;'>ParkPredict</div>
        <div style='font-size:0.75rem; color:#475569; letter-spacing:2px; text-transform:uppercase;'>AI Demand Forecaster</div>
    </div>
    <hr style='border-color:#1e2d3d; margin:12px 0 20px;'>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Event Details")

    event_type = st.selectbox("Event Type", EVENT_TYPES, index=0)
    venue      = st.selectbox("Venue", VENUES)
    city       = st.selectbox("City", CITIES)

    st.markdown("### 📅 Date & Time")
    month      = st.slider("Month", 1, 12, 6)
    day_of_week = st.selectbox("Day of Week",
                    options=[0,1,2,3,4,5,6],
                    format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    is_weekend  = int(day_of_week >= 5)
    is_holiday  = st.checkbox("Public Holiday?", value=False)

    st.markdown("### 👥 Attendees & Event Config")
    expected_attendees = st.number_input("Expected Attendees", 50, 10000, 500, 50)
    duration_hours     = st.selectbox("Duration (hours)", [2, 3, 4, 6, 8, 12], index=2)
    num_sessions       = st.slider("Number of Sessions", 1, 15, 3)
    registration_mode  = st.selectbox("Registration Mode", REG_MODES)
    is_paid_event      = int(st.checkbox("Paid Event?"))
    has_vip_guests     = int(st.checkbox("VIP / Chief Guests?"))
    alumni_batch_size  = st.slider("Alumni Batch Size (0 if not alumni event)", 0, 500, 0)

    st.markdown("### 🌤️ Weather & Environment")
    weather          = st.selectbox("Weather Condition", WEATHERS)
    temperature      = st.slider("Temperature (°C)", 5, 45, 28)
    is_rainy         = int(weather == "Rainy")

    st.markdown("### 🏗️ Infrastructure")
    total_parking_capacity          = st.number_input("Total Parking Capacity", 50, 5000, 500, 50)
    public_transport_score          = st.slider("Public Transport Score (1–10)", 1.0, 10.0, 5.0, 0.1)
    distance_from_city_center_km    = st.slider("Distance from City Center (km)", 0.5, 40.0, 5.0, 0.5)
    carpooling_promoted             = int(st.checkbox("Carpooling Promoted?"))

    st.markdown("### 🤖 Model Selection")
    selected_model = st.selectbox("Choose ML Model", list(all_models.keys()),
                                   index=list(all_models.keys()).index(best_model_name))

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚗 Predict Parking Demand", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding: 12px 0 4px;'>
  <span style='font-family:Syne; font-size:2.2rem; font-weight:800; color:#e2e8f0;'>Parking Demand Prediction</span><br>
  <span style='color:#64748b; font-size:0.95rem;'>Alumni Meet &amp; Conference Forecasting System &nbsp;|&nbsp; AI-Powered Analytics</span>
</div>
<hr style='border-color:#1e2d3d; margin: 16px 0 24px;'>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🎯  Predict", "📊  Dataset Insights", "🏆  Model Performance", "📖  Project Info"])


# ════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════
with tab1:
    input_data = {
        "event_type": event_type, "venue": venue, "city": city,
        "month": month, "day_of_week": day_of_week, "is_weekend": is_weekend,
        "is_holiday": int(is_holiday), "expected_attendees": expected_attendees,
        "duration_hours": duration_hours, "registration_mode": registration_mode,
        "is_paid_event": is_paid_event, "num_sessions": num_sessions,
        "has_vip_guests": has_vip_guests, "alumni_batch_size": alumni_batch_size,
        "weather": weather, "is_rainy": is_rainy,
        "temperature_celsius": temperature,
        "total_parking_capacity": total_parking_capacity,
        "public_transport_score": public_transport_score,
        "distance_from_city_center_km": distance_from_city_center_km,
        "carpooling_promoted": carpooling_promoted,
    }

    demand = predict(input_data, selected_model)
    demand_cat, alert_class, recommendation = demand_category(demand, total_parking_capacity)
    surplus_deficit = total_parking_capacity - demand
    utilization_pct = min(100, round(demand / total_parking_capacity * 100, 1))

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{demand:,}</div>
            <div class="metric-label">Predicted Demand</div>
            <div class="metric-sub">Vehicles expected</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{utilization_pct}%</div>
            <div class="metric-label">Utilization Rate</div>
            <div class="metric-sub">of total capacity</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        color = "#ef4444" if surplus_deficit < 0 else "#22c55e"
        sign  = "+" if surplus_deficit >= 0 else ""
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{color};">{sign}{surplus_deficit:,}</div>
            <div class="metric-label">{'Surplus' if surplus_deficit>=0 else 'Deficit'}</div>
            <div class="metric-sub">Available slots</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#a78bfa;">{selected_model.split()[0]}</div>
            <div class="metric-label">Model Used</div>
            <div class="metric-sub">{results_df[results_df['Model']==selected_model]['R2'].values[0]:.3f} R²</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Status alert
    st.markdown(f'<div class="{alert_class}"><b>{demand_cat}</b> — {recommendation}</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Gauges + comparisons
    col_l, col_r = st.columns([1, 1])

    with col_l:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=utilization_pct,
            delta={"reference": 70, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#22c55e"}},
            number={"suffix": "%", "font": {"size": 36, "color": "#e2e8f0"}},
            title={"text": "Parking Utilization", "font": {"size": 14, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"color": "#64748b"}},
                "bar": {"color": "#3b82f6"},
                "bgcolor": "#131f2e",
                "bordercolor": "#1e3a5f",
                "steps": [
                    {"range": [0, 70],  "color": "#0d2b18"},
                    {"range": [70, 90], "color": "#2a1f08"},
                    {"range": [90, 100],"color": "#2a0d0d"},
                ],
                "threshold": {"line": {"color": "#ef4444", "width": 3}, "value": 90},
            }
        ))
        fig_gauge.update_layout(
            height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", margin=dict(t=40, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_r:
        # Capacity breakdown bar
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["Total Capacity", "Predicted Demand", "Available"],
            y=[total_parking_capacity, demand, max(0, surplus_deficit)],
            marker_color=["#3b82f6", "#f59e0b", "#22c55e"],
            text=[total_parking_capacity, demand, max(0, surplus_deficit)],
            textposition="outside",
            textfont={"color": "#e2e8f0"},
            width=0.5,
        ))
        fig_bar.update_layout(
            title="Capacity vs Demand Breakdown",
            height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis={"gridcolor": "#1e2d3d", "tickfont": {"color": "#94a3b8"}},
            yaxis={"gridcolor": "#1e2d3d", "tickfont": {"color": "#94a3b8"}},
            margin=dict(t=40, b=10, l=10, r=10),
            title_font={"color": "#94a3b8", "size": 13},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # All-models comparison
    if predict_btn or True:
        st.markdown('<div class="section-header">🔬 All Models Comparison</div>', unsafe_allow_html=True)
        model_preds = {name: predict(input_data, name) for name in all_models}
        comp_df = pd.DataFrame(list(model_preds.items()), columns=["Model", "Predicted Demand"])
        comp_df = comp_df.sort_values("Predicted Demand")

        fig_comp = px.bar(
            comp_df, x="Predicted Demand", y="Model", orientation="h",
            color="Predicted Demand", color_continuous_scale=["#1e3a5f", "#3b82f6", "#06b6d4"],
            text="Predicted Demand",
        )
        fig_comp.update_traces(textfont_color="#e2e8f0", textposition="outside")
        fig_comp.update_layout(
            height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", coloraxis_showscale=False,
            xaxis={"gridcolor": "#1e2d3d", "tickfont": {"color": "#94a3b8"}},
            yaxis={"gridcolor": "#1e2d3d", "tickfont": {"color": "#94a3b8"}},
            margin=dict(t=10, b=10, l=10, r=60),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Attendees sensitivity
    st.markdown('<div class="section-header">📈 Sensitivity Analysis — Attendees vs Demand</div>', unsafe_allow_html=True)
    attendee_range = list(range(100, min(expected_attendees * 2, 5001), 50))
    sens_preds = []
    for att in attendee_range:
        inp = input_data.copy(); inp["expected_attendees"] = att
        sens_preds.append(predict(inp, selected_model))

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=attendee_range, y=sens_preds,
        mode="lines", line={"color": "#3b82f6", "width": 3},
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        name="Predicted Demand"
    ))
    fig_sens.add_vline(x=expected_attendees, line_color="#f59e0b", line_dash="dash",
                       annotation_text=f"  Current: {expected_attendees}", annotation_font_color="#f59e0b")
    fig_sens.update_layout(
        height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", showlegend=False,
        xaxis={"title": "Expected Attendees", "gridcolor": "#1e2d3d", "tickfont": {"color": "#94a3b8"}},
        yaxis={"title": "Parking Demand", "gridcolor": "#1e2d3d", "tickfont": {"color": "#94a3b8"}},
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    # Recommendations
    st.markdown('<div class="section-header">💡 Smart Recommendations</div>', unsafe_allow_html=True)
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.info(f"""
**🚗 Parking Management**
- Allocate **{demand:,} spots** minimum
- Reserve **{max(0, int(demand*0.1)):,}** slots for VIPs/disabled
- Deploy **{max(1, demand//50)}** traffic marshals
- Stagger entry/exit with **{max(2, num_sessions)}** time slots
        """)
    with rec_col2:
        st.info(f"""
**🚌 Overflow & Alternatives**
- Overflow capacity needed: **{max(0, demand - total_parking_capacity):,}** spots
- Arrange **{max(1, (demand-total_parking_capacity)//50 if demand>total_parking_capacity else 0)}** shuttle buses from off-site
- Promote carpooling → saves **~{int(demand*0.15):,}** spaces
- Partner with nearby lots within **2km**
        """)


# ════════════════════════════════════════
# TAB 2 — DATASET INSIGHTS
# ════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, (val, lbl) in zip([m1, m2, m3, m4], [
        (f"{len(df):,}", "Total Records"),
        (f"{int(df['parking_demand'].mean()):,}", "Avg Demand"),
        (f"{int(df['parking_demand'].max()):,}", "Peak Demand"),
        (f"{df['event_type'].nunique()}", "Event Types"),
    ]):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig = px.box(df, x="event_type", y="parking_demand", color="event_type",
                     title="Parking Demand by Event Type",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", showlegend=False, height=380,
                          xaxis_tickangle=-30,
                          xaxis={"gridcolor":"#1e2d3d","tickfont":{"color":"#94a3b8"}},
                          yaxis={"gridcolor":"#1e2d3d","tickfont":{"color":"#94a3b8"}},
                          title_font={"color":"#94a3b8"}, margin=dict(t=40,b=60))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        avg_month = df.groupby("month")["parking_demand"].mean().reset_index()
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        avg_month["month_name"] = avg_month["month"].apply(lambda x: month_names[x-1])
        fig = px.bar(avg_month, x="month_name", y="parking_demand",
                     title="Average Demand by Month",
                     color="parking_demand", color_continuous_scale=["#1e3a5f","#3b82f6","#38bdf8"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", coloraxis_showscale=False, height=380,
                          xaxis={"gridcolor":"#1e2d3d","tickfont":{"color":"#94a3b8"}},
                          yaxis={"gridcolor":"#1e2d3d","title":"Avg Parking Demand","tickfont":{"color":"#94a3b8"}},
                          title_font={"color":"#94a3b8"}, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(df, x="expected_attendees", y="parking_demand",
                         color="event_type", opacity=0.6,
                         title="Attendees vs Parking Demand",
                         trendline="ols",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", height=380,
                          xaxis={"gridcolor":"#1e2d3d","tickfont":{"color":"#94a3b8"}},
                          yaxis={"gridcolor":"#1e2d3d","tickfont":{"color":"#94a3b8"}},
                          title_font={"color":"#94a3b8"}, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        weather_avg = df.groupby("weather")["parking_demand"].mean().reset_index()
        fig = px.pie(weather_avg, names="weather", values="parking_demand",
                     title="Avg Demand Distribution by Weather",
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     hole=0.4)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", height=380,
                          title_font={"color":"#94a3b8"}, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    st.markdown('<div class="section-header">🗺️ Day × Month Heatmap</div>', unsafe_allow_html=True)
    pivot = df.pivot_table(values="parking_demand", index="day_of_week", columns="month", aggfunc="mean")
    pivot.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig_heat = px.imshow(pivot, color_continuous_scale="Blues",
                         title="Average Parking Demand: Day of Week × Month")
    fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", height=320,
                           title_font={"color":"#94a3b8"}, margin=dict(t=40,b=10))
    st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("📋 View Raw Dataset (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)


# ════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🏆 Model Comparison Report</div>', unsafe_allow_html=True)

    styled_results = results_df.copy()
    st.dataframe(
        styled_results.style
        .background_gradient(subset=["R2"], cmap="Blues")
        .background_gradient(subset=["MAE","RMSE"], cmap="Reds_r")
        .format({"R2": "{:.4f}", "MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE(%)": "{:.2f}"}),
        use_container_width=True, height=260
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(results_df.sort_values("R2"), x="R2", y="Model", orientation="h",
                     title="R² Score (higher = better)", color="R2",
                     color_continuous_scale=["#1e3a5f","#3b82f6","#06b6d4"], text="R2")
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside", textfont_color="#e2e8f0")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", coloraxis_showscale=False, height=350,
                          xaxis={"gridcolor":"#1e2d3d","tickfont":{"color":"#94a3b8"}},
                          yaxis={"gridcolor":"#1e2d3d","tickfont":{"color":"#94a3b8"}},
                          title_font={"color":"#94a3b8"}, margin=dict(t=40,b=10,r=60))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(results_df.sort_values("MAE", ascending=False), x="MAE", y="Model", orientation="h",
                     title="MAE (lower = better)", color="MAE",
                     color_continuous_scale=["#064e3b","#f59e0b","#ef4444"], text="MAE")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", textfont_color="#e2e8f0")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", coloraxis_showscale=False, height=350,
                          xaxis={"gridcolor":"#1e2d3d","tickfont":{"color":"#94a3b8"}},
                          yaxis={"gridcolor":"#1e2d3d","tickfont":{"color":"#94a3b8"}},
                          title_font={"color":"#94a3b8"}, margin=dict(t=40,b=10,r=60))
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    if fi_df is not None:
        st.markdown('<div class="section-header">🔍 Feature Importance (Best Model)</div>', unsafe_allow_html=True)
        fig_fi = px.bar(fi_df.head(12), x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale=["#1e3a5f","#3b82f6","#38bdf8"])
        fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font_color="#e2e8f0", coloraxis_showscale=False, height=380,
                             xaxis={"gridcolor":"#1e2d3d","tickfont":{"color":"#94a3b8"}},
                             yaxis={"gridcolor":"#1e2d3d","tickfont":{"color":"#94a3b8"}},
                             margin=dict(t=10,b=10,r=30))
        st.plotly_chart(fig_fi, use_container_width=True)


# ════════════════════════════════════════
# TAB 4 — PROJECT INFO
# ════════════════════════════════════════
with tab4:
    st.markdown("""
<div class="section-header">📖 Project Overview</div>

<div style="color:#cbd5e1; line-height:1.9; font-size:0.95rem;">

**Parking Demand Prediction for Alumni Meet and Conferences** is a machine learning project that
forecasts the number of vehicles expected to require parking at large institutional events. Accurate
predictions help event organizers allocate resources, prevent overflow, and improve the experience
for thousands of attendees.

</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
#### 🎯 Objectives
- Predict parking demand based on event, environmental, and infrastructure features
- Compare multiple ML regression algorithms
- Provide actionable recommendations to organizers
- Enable sensitivity analysis for planning scenarios

#### 📦 Dataset Features (21 inputs)
| Category | Features |
|---|---|
| **Event** | Type, Venue, City, Sessions, VIP, Paid |
| **Temporal** | Month, Day, Weekend, Holiday |
| **Attendees** | Expected count, Alumni batch, Registration |
| **Weather** | Condition, Temperature, Rainfall |
| **Infrastructure** | Capacity, Transport score, Distance, Carpooling |
        """)

    with col2:
        st.markdown(f"""
#### 🤖 ML Models Used
| Model | R² Score |
|---|---|
""" + "\n".join([f"| {'⭐ ' if r['Model']==best_model_name else ''}{r['Model']} | {r['R2']:.4f} |"
                  for _, r in results_df.iterrows()]) + """

#### 🔬 Methodology
1. **Data Generation** — 2,000 realistic synthetic samples
2. **Preprocessing** — Label encoding + Standard scaling
3. **Training** — 80/20 train-test split, 6 model comparison
4. **Evaluation** — MAE, RMSE, R², MAPE metrics
5. **Deployment** — Streamlit interactive app
        """)

    st.markdown("""
#### 📁 Project Structure
```
parking_prediction/
├── data/
│   └── parking_dataset.csv          # 2000-sample synthetic dataset
├── models/
│   └── parking_model.pkl            # Trained models + encoders + scaler
├── utils/
│   ├── data_generator.py            # Synthetic data generation
│   └── model_trainer.py             # Model training & evaluation pipeline
├── app.py                           # Streamlit application (this file)
├── requirements.txt                 # Dependencies
└── README.md                        # Project documentation
```
    """)

    st.markdown("""
#### 🚀 How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (auto-generates data)
python utils/model_trainer.py

# 3. Launch the app
streamlit run app.py
```
    """)
