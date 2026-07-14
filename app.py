from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# ────────────────────────────────────────────────────────────────────────────
# Page Configuration & Global Branding
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RiskShield — Motor Underwriting Suite",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Open Sans typography + Custom Enterprise Styling (Aligned with Lecturer Blueprint)
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght=400;600;700&display=swap');
      html, body, [class*="css"]  { font-family: 'Open Sans', sans-serif; }
      .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
      
      /* Card Metrics styling */
      [data-testid="stMetric"] {
          background: rgba(30, 58, 138, 0.04);
          border: 1px solid rgba(30, 58, 138, 0.1);
          padding: 14px 18px; border-radius: 12px;
      }
      [data-testid="stMetricLabel"] { font-weight: 600; color: #1E3A8A; }
      
      /* Hero Banner styling */
      .hero {
          background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #3B82F6 100%);
          color: white; padding: 28px 32px; border-radius: 16px;
          margin-bottom: 20px;
          box-shadow: 0 4px 15px rgba(30, 58, 138, 0.15);
      }
      .hero h1 { color: white; margin: 0; font-weight: 700; }
      .hero p  { color: rgba(255,255,255,0.85); margin: 6px 0 0 0; font-size: 1.05rem; }
      
      /* Underwriter System action card block */
      .verdict-card {
          background: #FFFFFF;
          padding: 20px;
          border-radius: 12px;
          box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
          border: 1px solid #E2E8F0;
          margin-bottom: 15px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# Global Scoped Portfolio Summary Statistics
# ────────────────────────────────────────────────────────────────────────────
actual_means = [7.41, 15.20, 22.10, 35.50, 48.90, 62.10, 78.40, 99.80, 120.50, 160.73]
portfolio_avg = 65.00
deciles = list(range(1, 11))

# ────────────────────────────────────────────────────────────────────────────
# Application Header
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <h1>🚗 RiskShield | Motor Underwriting Intelligence</h1>
      <p>SMM284 Group Report · Champion Poisson GLM Production Deployment Engine</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Informational Modal Dialog
@st.dialog("About this app")
def show_about():
    st.markdown("""
**Underwriting Context & Stack**

This dashboard productizes our group's modeling pipeline for the **Chief Underwriting Officer**.
It enables fast-tracking renewals, auditing anomalous risks, and highlighting model limitations.

*   **Champion Model:** Poisson GLM (Optimal CV MAE & strict transparency).
*   **Interactive Tabs:** Navigate through exploratory analytics, model validation performance, and live inference testing.
*   **Group 06 Members:** Abdulrahman Alolyan, Chhavi Ahuja, & Tracy Rotich.
    """)

col_h1, col_h2 = st.columns([5, 1])
with col_h2:
    if st.button("ℹ️ About System", use_container_width=True):
        show_about()

# ────────────────────────────────────────────────────────────────────────────
# Sidebar Panel Layout
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### **Control Center**")
    st.info("🔒 Authorized Underwriter Access Only.")
    st.divider()
    st.caption("Built for Applied ML · Bayes Business School")

# Enterprise KPI Dashboard Ribbon
m1, m2, m3, m4 = st.columns(4)
m1.metric("Database Transactions", "105,555")
m2.metric("Portfolio Average Claim", "$65.00")
m3.metric("Top Decile Risk Multiplier", "2.5x")
m4.metric("Deployed Framework", "Poisson GLM")

# ────────────────────────────────────────────────────────────────────────────
# Main Application Portals (Tabs Layout)
# ────────────────────────────────────────────────────────────────────────────
tab_briefing, tab_explore, tab_compare, tab_predict = st.tabs(
    ["🏢 Executive Briefing", "📊 Portfolio Insights", "🎯 Model Champions", "🧮 Underwriting Calculator"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Executive Briefing
# ════════════════════════════════════════════════════════════════════════════
with tab_briefing:
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("### **Underwriting Strategy & Core Proposition**")
        st.write(
            "At each annual renewal, the insurer must set a premium for every motor policy. "
            "If our calculations are incorrect, we generate direct underwriting losses. "
            "While individual losses are highly random, our Poisson GLM model successfully ranks policies "
            "from safest to riskiest, beating a flat baseline pricing structure."
        )
        st.markdown(
            "> **Key Discovery:** Sorting renewals into predicted risk deciles reveals that actual claim "
            "costs climb systematically. The safest decile runs at a tenth of the baseline average, while the "
            "riskiest decile runs at **2.5x the portfolio average** ($160.73 vs $65.00)."
        )
    with col_right:
        st.markdown(
            """
            <div class="verdict-card" style="border-left: 5px solid #1E3A8A;">
                <h4 style="color: #1E3A8A; margin-top: 0;">💡 Actionable CUO Directives</h4>
                <ul style="padding-left: 1.2rem; line-height: 1.6;">
                    <li><strong>Deploy Ordinal Ranking:</strong> Feed decile rankings directly to API handlers to automate low-risk renewals (Deciles 1–4).</li>
                    <li><strong>Enforce Auditing:</strong> Flag all renewals falling into Deciles 9 and 10 for manual, senior underwriting referral.</li>
                    <li><strong>Postpone Absolute Pricing:</strong> Recalibrate baseline predicted totals before adjusting commercial prices to account for late-reported claim immaturity.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Portfolio Insights
# ════════════════════════════════════════════════════════════════════════════
with tab_explore:
    view = st.segmented_control(
        "Explore Sub-view",
        options=["Target Claim Skew", "Categorical Driver Trends", "Numeric Features"],
        default="Target Claim Skew",
    )

    if view == "Target Claim Skew":
        col_l, col_r = st.columns([2, 3])
        with col_l:
            st.markdown("#### **Extremely Right-Skewed Losses**")
            st.write(
                "Over 98% of policies record exactly **$0 in claims** in any given policy year. "
                "Because of this extreme skew, raw regression evaluations like R² are unhelpful. "
                "Our optimization focuses entirely on Mean Absolute Error (MAE)."
            )
        with col_r:
            fig = go.Figure(go.Bar(
                x=["No Claims (Zero Loss)", "Incurred Active Claims"],
                y=[98.5, 1.5],
                marker_color=["#1E3A8A", "#EF4444"],
                text=["98.5%", "1.5%"],
                textposition='auto',
                width=0.4
            ))
            fig.update_layout(height=300, yaxis_title="Percentage of Portfolio (%)", margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    elif view == "Categorical Driver Trends":
        selected_cat = st.pills("Vehicle Category", ["Type 1 (Motorcycle)", "Type 2 (Private)", "Type 3 (Commercial)", "Type 4 (Fleet)"], default="Type 2 (Private)")
        cat_costs = {"Type 1 (Motorcycle)": 45, "Type 2 (Private)": 135, "Type 3 (Commercial)": 210, "Type 4 (Fleet)": 340}
        
        fig = go.Figure(go.Bar(
            x=list(cat_costs.keys()), y=list(cat_costs.values()),
            marker_color=["#F59E0B" if k == selected_cat else "#3B82F6" for k in cat_costs.keys()]
        ))
        fig.update_layout(title=f"Claim Severity Highlights: {selected_cat}", yaxis_title="Average Claim Cost ($)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Numeric Features":
        age_bands = ["17-25", "25-35", "35-45", "45-55", "55-65", "65-75", "75+"]
        mean_costs_age = [320, 180, 140, 110, 95, 85, 130]
        fig_age = px.bar(
            x=age_bands, y=mean_costs_age,
            labels={"x": "Driver Age Group", "y": "Mean Cost ($)"},
            title="<b>U-Shaped Age Risk Curve across Portfolio</b>",
            color_discrete_sequence=["#1E3A8A"]
        )
        fig_age.update_layout(height=350)
        st.plotly_chart(fig_age, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Champions
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### **Model Tournament Comparisons**")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("##### **5-Fold Group Cross-Validation (2015-2017)**")
        cv_df = pd.DataFrame({
            "Model Name": ["Poisson GLM (Champion)", "Random Forest (Shallow)", "XGBoost (Shallow)", "Baseline (Mean)", "Random Forest (Deeper)", "XGBoost (Deeper)"],
            "CV MAE ($)": [308.12, 312.20, 321.05, 324.50, 313.80, 327.80],
            "CV RMSE ($)": [1210.4, 1225.1, 1250.3, 1280.0, 1238.9, 1312.0],
            "State": ["Optimal", "Robust", "Overfit Risk", "Uninformative", "Overfitted", "Overfitted"]
        })
        st.dataframe(cv_df.style.highlight_min(subset=["CV MAE ($)"], color="#DCFCE7"))
        st.caption("Folds are strictly split by Customer ID to eliminate leakage.")

    with col_t2:
        st.markdown("##### **2018 Unseen Holdout Evaluation**")
        test_df = pd.DataFrame({
            "Model Name": ["Poisson GLM", "Random Forest (Shallow)", "XGBoost (Shallow)", "Baseline"],
            "Test MAE ($)": [141.20, 143.50, 149.10, 148.90],
            "Volume Overprediction": ["3.19x", "3.08x", "3.22x", "1.00x"]
        })
        st.dataframe(test_df.style.highlight_min(subset=["Test MAE ($)"], color="#DCFCE7"))
        st.caption("3.19x overprediction is caused by structural data maturity delay in 2018 records.")

    st.divider()
    st.markdown("#### **Interactive Risk Decile Lift Chart**")
    
    fig_lift = go.Figure()
    fig_lift.add_trace(go.Bar(
        x=deciles, y=actual_means,
        marker=dict(color=["#10B981" if x <= 4 else "#F59E0B" if x <= 8 else "#EF4444" for x in deciles]),
        name="Actual Decile Average"
    ))
    fig_lift.add_shape(
        type="line", line=dict(color="#64748B", width=2, dash="dash"),
        x0=0.5, x1=10.5, y0=portfolio_avg, y1=portfolio_avg,
        name="Portfolio Average Baseline"
    )
    fig_lift.update_layout(
        xaxis=dict(title="Model Predicted Decile", tickmode="linear"),
        yaxis=dict(title="Actual Mean Historical Claim Cost ($)"),
        height=380,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_lift, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Underwriting Calculator (Genuine Machine Learning Inference Fragment)
# ════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.caption("Adjust policyholder metrics on the fly. Calculations execute directly inside the loaded serialized pipeline script.")

    # Safely resource-cache the pipeline binaries from GitHub directory
    @st.cache_resource(show_spinner="Loading scoring engine layers...")
    def load_scoring_model():
        try:
            return joblib.load("champion_poisson_model.pkl")
        except FileNotFoundError:
            return None

    trained_model = load_scoring_model()

    if trained_model is None:
        st.error("❌ **`champion_poisson_model.pkl` file missing from Repository.** Please upload the notebook-exported pickle file into your GitHub folder to initialize live scoring.")
    else:
        @st.fragment
        def prediction_fragment():
            """Encapsulated UI segment ensuring responsive real-time data input processing."""
            
            with st.container(border=True):
                st.markdown("#### **Policyholder Input Matrix**")
                col_u1, col_u2, col_u3 = st.columns(3)
                
                with col_u1:
                    st.markdown("**👤 Demographics**")
                    driver_age = st.slider("Driver Age (Age)", 17, 95, 35, key="calc_age")
                    licence_years = st.slider("Years Licensed (Licence_years)", 0, 75, 15, key="calc_lic")
                    customer_years = st.slider("Insurer Tenure (Customer_years)", 0.0, 25.0, 4.0, step=0.5, key="calc_ten")
                    
                with col_u2:
                    st.markdown("**🚗 Vehicle Parameters**")
                    vehicle_age = st.slider("Vehicle Age (Vehicle_age)", 0, 30, 6, key="calc_v_age")
                    power = st.number_input("Engine Power (Power)", min_value=10, max_value=800, value=110, key="calc_pow")
                    cylinder = st.number_input("Cylinder Capacity (cc)", min_value=100, max_value=8000, value=1600, key="calc_cyl")
                    length = st.number_input("Vehicle Length (mm)", min_value=1000, max_value=6000, value=4200, key="calc_len")
                    weight = st.number_input("Vehicle Weight (kg)", min_value=300, max_value=4000, value=1300, key="calc_weight")
                    doors = st.selectbox("Door Count (N_doors)", ["4", "2", "3", "5"], key="calc_doors")
                    
                with col_u3:
                    st.markdown("**📄 Operational Categories**")
                    risk_type = st.selectbox("Underwriting Classification (Type_risk)", ["2", "1", "3", "4"], key="calc_risk")
                    fuel_type = st.selectbox("Fuel Type (Type_fuel)", ["Gasoline", "Diesel", "LPG", "Unknown"], key="calc_fuel")
                    area = st.selectbox("Area Code (Area)", ["1", "2", "3", "4", "5"], key="calc_area")
                    channel = st.selectbox("Distribution Channel", ["1", "2", "3"], key="calc_chan")
                    payment = st.selectbox("Payment Method", ["1", "2", "3"], key="calc_pay")
                    second_driver = st.selectbox("Second Driver Registry", ["0", "1"], key="calc_sec")
                    policies_in_force = st.number_input("Policies in Force", 1, 10, 1, key="calc_pif")

            # Package features into structural format matching notebook ColumnTransformer layout
            input_row = pd.DataFrame([{
                "Age": float(driver_age),
                "Licence_years": float(licence_years),
                "Vehicle_age": float(vehicle_age),
                "Customer_years": float(customer_years),
                "Power": float(power),
                "Cylinder_capacity": float(cylinder),
                "Length": float(length),
                "Weight": float(weight),
                "Policies_in_force": float(policies_in_force),
                "Length_missing": 0,
                "Type_risk": str(risk_type),
                "Type_fuel": str(fuel_type),
                "Area": str(area),
                "Distribution_channel": str(channel),
                "Payment": str(payment),
                "Second_driver": str(second_driver),
                "N_doors": str(doors)
            }])

            # Execute explicit live pipeline inference pass
            real_predicted_cost = trained_model.predict(input_row)[0]
            
            # Print exact raw mathematical response
            st.write(f"🔬 *Real-Time Mathematical Inference:* **Estimated Base Claim Cost = `${real_predicted_cost:.2f}`**")

            # Map mathematical predictions safely into empirical portfolio sorting deciles
            if real_predicted_cost < 20.0: decile = 1
            elif real_predicted_cost < 40.0: decile = 2
            elif real_predicted_cost < 60.0: decile = 3
            elif real_predicted_cost < 80.0: decile = 4
            elif real_predicted_cost < 100.0: decile = 5
            elif real_predicted_cost < 125.0: decile = 6
            elif real_predicted_cost < 150.0: decile = 7
            elif real_predicted_cost < 185.0: decile = 8
            elif real_predicted_cost < 240.0: decile = 9
            else: decile = 10

            rel_risk = actual_means[decile-1] / portfolio_avg

            st.markdown("### **Calculated System Verdict**")
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric("Assigned Risk Bracket", f"Decile {decile} / 10")
                
            with col_res2:
                st.metric("Expected Premium Multiplier", f"{rel_risk:.2f}x", "vs Portfolio Mean Base")
                
            with col_res3:
                if decile <= 4:
                    status, color, desc = "AUTO-PASS", "#10B981", "Risk metrics are within optimal baselines. Fast-track automated rate."
                elif decile <= 8:
                    status, color, desc = "STANDARD AUDIT", "#F59E0B", "Manually evaluate traditional pricing loading factors."
                else:
                    status, color, desc = "REFER TO SENIOR CUO", "#EF4444", "Severe systemic claim probability. Route for manual premium adjustments."
                    
                st.markdown(
                    f"""
                    <div class="verdict-card" style="border-top: 4px solid {color}; padding: 10px 15px;">
                        <p style="margin: 0; font-weight: 700; color: {color};">{status}</p>
                        <p style="margin: 3px 0 0 0; font-size: 0.85rem; color: #475569;">{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Highlighting placement over background distributions
            fig_ind = go.Figure()
            fig_ind.add_trace(go.Bar(x=deciles, y=actual_means, name="Portfolio Baseline", marker=dict(color=["#E2E8F0"] * 10)))
            fig_ind.add_trace(go.Bar(x=[decile], y=[actual_means[decile-1]], name="Applicant", marker=dict(color=["#1E3A8A" if decile <= 8 else "#EF4444"], line=dict(color="#0F172A", width=1.5))))
            fig_ind.update_layout(
                title="<b>Applicant Placement on Empirical Loss Curve</b>",
                xaxis=dict(title="Risk Decile Sorting Band", tickmode="linear"),
                yaxis=dict(title="Empirical Mean Class Cost ($)"),
                height=250,
                showlegend=False,
                margin=dict(t=30, b=10)
            )
            st.plotly_chart(fig_ind, use_container_width=True)

        prediction_fragment()
