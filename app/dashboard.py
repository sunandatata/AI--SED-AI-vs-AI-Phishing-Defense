# app/dashboard.py
"""
AI²-SED Cinematic Dashboard v3
Animated transitions + PDF export capability
"""

from __future__ import annotations
import os, io
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# === COLORS ===
ACCENT = "#00fff2"
ACCENT2 = "#ff00c8"
TEXT = "#e8f0ff"

def render_dashboard():
    ROOT = Path(__file__).resolve().parents[1]
    SYN_DIR = ROOT / "data" / "synthetic"
    BASELINE = ROOT / "blue_agent" / "models" / "baseline.joblib"
    ROBERTA = ROOT / "blue_agent" / "models" / "roberta" / "final"
    T5 = ROOT / "red_agent" / "models" / "t5_phish"

    # === PAGE STYLE ===
    st.markdown("""
    <style>
    body, .stApp {background: radial-gradient(circle at 10% 10%, #08121a 0%, #000 90%);}
    .slide-in {
        animation: slideIn 1s ease;
    }
    @keyframes slideIn {
        0% {opacity: 0; transform: translateX(40px);}
        100% {opacity: 1; transform: translateX(0);}
    }
    .card {
        background:rgba(255,255,255,0.05);
        border-radius:18px;
        padding:18px 20px;
        backdrop-filter:blur(8px);
        box-shadow:0 0 25px rgba(0,255,242,0.08);
        transition:all 0.4s ease-in-out;
    }
    .card:hover {transform:translateY(-6px); box-shadow:0 0 35px rgba(0,255,242,0.15);}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class='slide-in' style='text-align:center;color:#00fff2;'>✨ AI²-SED Motion Dashboard ✨</h2>", unsafe_allow_html=True)
    st.caption("Real-time visualization • Research presentation-ready")

    # === LOAD DATA ===
    if not SYN_DIR.exists() or not list(SYN_DIR.glob("round_*.csv")):
        st.info("No synthetic rounds yet. Run at least one.")
        return

    rounds = sorted(SYN_DIR.glob("round_*.csv"), key=os.path.getmtime)
    data = []
    for r in rounds:
        try:
            df = pd.read_csv(r)
            miss = (df["blue_label"] == 0).sum()
            dei = round(100 - (miss / max(1, len(df))) * 100, 2)
            data.append({"Round": int(r.stem.split("_")[-1]), "Samples": len(df), "DEI": dei})
        except Exception:
            continue

    meta = pd.DataFrame(data).sort_values("Round")
    latest_dei = meta["DEI"].iloc[-1]

    # === ANIMATED COUNT-UP METRIC ===
    st.markdown("<div class='card slide-in'>", unsafe_allow_html=True)
    placeholder = st.empty()
    for val in range(0, int(latest_dei)+1, 2):
        placeholder.metric("Current DEI", f"{val:.0f}%")
        time_ms = 0.02
        import time; time.sleep(time_ms)
    placeholder.metric("Current DEI", f"{latest_dei:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    # === DEI GAUGE ===
    st.markdown("<div class='card slide-in'>", unsafe_allow_html=True)
    st.markdown("### 🧠 Defensive Effectiveness Index (Gauge)")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_dei,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': ACCENT},
            'steps': [
                {'range': [0, 60], 'color': '#3a0c12'},
                {'range': [60, 85], 'color': '#122427'},
                {'range': [85, 100], 'color': '#0b2f31'},
            ],
            'threshold': {'line': {'color': ACCENT2, 'width': 4}, 'value': latest_dei}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # === DEI TREND CHART ===
    st.markdown("<div class='card slide-in'>", unsafe_allow_html=True)
    st.markdown("### 📈 DEI Trend Over Rounds")
    fig2 = px.line(meta, x="Round", y="DEI", markers=True)
    fig2.update_traces(line=dict(color=ACCENT, width=3), marker=dict(size=10, color=ACCENT2))
    fig2.update_layout(transition_duration=800, plot_bgcolor="rgba(255,255,255,0.03)",
                       paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # === TABLE ===
    st.markdown("<div class='card slide-in'>", unsafe_allow_html=True)
    st.markdown("### 🧩 Round Summary Table")
    st.dataframe(meta.style.background_gradient(cmap="Blues").highlight_max(subset=["DEI"], color="#00fff255"))
    st.markdown("</div>", unsafe_allow_html=True)

    # === PDF EXPORT ===
