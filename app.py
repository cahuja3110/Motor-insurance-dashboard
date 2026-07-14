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

# 🎨 TRANSFORMATION: Modern Corporate Theme Styling
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700&display=swap');
      
      html, body, [class*="css"] { 
          font-family: 'Inter', sans-serif; 
          background-color: #F8FAFC; 
      }
      
      .main .block-container { 
          padding-top: 1.5rem; 
          padding-bottom: 2rem; 
          max-width: 1200px; 
      }

      .hero {
          background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
          color: white; 
          padding: 35px 40px; 
          border-radius: 16px;
          margin-bottom: 25px;
          box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.08);
      }
      .hero h1 { color: white !important; font-weight: 700 !important; font-size: 2.2rem !important; }
      .hero p  { color: #94A3B8; margin-top: 8px; font-size: 1.1rem; }

      [data-testid="stMetric"] {
          background: #FFFFFF;
          border: 1px solid #E2E8F0;
          padding: 16px 20px; 
          border-radius: 12px;
          box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);
      }
      [data-testid="stMetricLabel"] { font-weight: 600; color: #1E3A8A; }

      .verdict-card {
          background: #FFFFFF;
          padding: 16px 20px;
          border-radius: 12px;
          box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
          border-left: 5px solid #E2E8F0;
          margin-bottom: 15px;
      }

      div[data-testid="stTabs"] {
          background-color: #FFFFFF;
          border-radius: 12px;
          padding: 8px 16px;
          box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);
          margin-bottom: 25px;
      }
      
      button[data-baseweb="tab"] {
          font-family: 'Inter', sans-serif !important;
          font-size: 0.95rem !important;
          font-weight: 600 !important;
          color: #64748B !important; 
          padding: 12px 20px !important;
          border-bottom: 2px solid transparent !important;
          transition: all 0.2s ease-in-out !important;
      }

      button[data-baseweb="tab"]:hover {
          color: #1E3A8A !important; 
          background-color: #F1F5F9 !important;
          border-radius: 8px !important;
       }

      button[data-baseweb="tab"][aria-selected="true"] {
          color: #1E3A8A !important;
          border-bottom: 3px solid #1E3A8A !important; 
          background-color: transparent !important;
      }

      [data-testid="stTable"] {
          background: #FFFFFF;
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

actual_means = [7.41, 15.20, 22.10, 35.50, 48.90, 62.10, 78.40, 99.80, 120.50, 160.73]
portfolio_avg = 65.00
deciles = list(range(1, 11))

# Hero Header Module with updated exact requested string
st.markdown(
    """
    <div class="hero">
      <h1>🚗 RiskShield | Motor Underwriting Intelligence</h1>
      <p>SMM284 Group 06 · Champion Poisson GLM Production Deployment Engine</p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.dialog("About this app")
def show_about():
    st.markdown("""
**Underwriting Context & Tech Stack**
This dashboard productizes our group's modeling pipeline for the **Chief Underwriting Officer**.
- **Champion Model:** Poisson GLM (Trained on 100k+ policy transactions)
- **Group 06 Members:** Abdulrahman Alolyan, Chhavi Ahuja, Dufie Denteh Priscilla, & Tracy Rotich.
- **Academic Environment:** Bayes Business School (Applied Machine Learning).
    """)

col_h1, col_h2 = st.columns([5, 1])
with col_h2:
    if st.button("ℹ️ About System", use_container_width=True):
        show_about()

with st.sidebar:
    st.markdown("### **Control Center**")
    st.info("🔒 Authorized Underwriter Access Only.")
    st.divider()
    st.caption("Built for Applied ML · Bayes Business School")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Database Transactions", "105,555")
m2.metric("Portfolio Average Claim", "$65.00")
m3.metric("Top Decile Risk Multiplier", "2.47x")
m4.metric("Deployed Framework", "Poisson GLM")

tab_briefing, tab_explore, tab_compare, tab_predict = st.tabs(
    ["🏢 Executive Briefing", "📊 Portfolio Insights & Trends", "🎯 Model Champions", "🧮 Underwriting Calculator"]
)

# ────────────────────────────────────────────────────────────────────────────
# TAB 1: EXECUTIVE BRIEFING (UPDATED FROM GENERIC TO STRUCTURAL USER OVERVIEW)
# ────────────────────────────────────────────────────────────────────────────
with tab_briefing:
    st.markdown("### **System Blueprint & Operational Purpose**")
    st.markdown(
        "Welcome to the **RiskShield Production Workspace**. This interface bridges the gap between raw "
        "actuarial data science and live commercial distribution. The purpose of this system is to deploy our team's "
        "validated machine learning pipelines directly into the hands of underwriters to eliminate flat baseline pricing."
    )
    
    with st.container(border=True):
        st.markdown("#### **🛠️ How to Navigate the Architecture**")
        st.markdown(
            """
            1. **📊 Portfolio Insights:** Review the exploratory trends, exposure histories, and zero-inflation dynamics within our macro risk pools.
            2. **🎯 Model Champions:** Audit the mathematical tournament matrices. This page visualizes how our champion pipeline increases business efficiency over simple linear models.
            3. **🧮 Underwriting Calculator:** Input live customer matrices. The system parses inputs through the serialized model code to output dynamic loading factors.
            """
        )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        with st.container(border=True):
            st.markdown("#### **🔴 Historical Baseline Inefficiencies**")
            st.write("Flat average premium engines mask the true variation in user hazards, triggering adverse portfolio performance.")
    with col_b2:
        with st.container(border=True):
            st.markdown("#### **🟢 Poisson GLM Strategic Advantage**")
            st.write("Our champion framework isolates volatile extreme-risk exposures from optimal segments, ensuring strict competitive advantages.")

# ────────────────────────────────────────────────────────────────────────────
# TAB 2: PORTFOLIO INSIGHTS & HISTORICAL TRENDS (EDA RESTORED WITH PLOTLY TRENDS)
# ────────────────────────────────────────────────────────────────────────────
with tab_explore:
    st.markdown("### **Exploratory Data Analysis (EDA) & Portfolio Characterization**")
    st.markdown("This workspace visualizes the underlying trends and distribution characteristics extracted from your motor insurance registry.")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        with st.container(border=True):
            st.markdown("#### **📈 Historical Portfolio Claim Frequency Trend**")
            st.write("Tracking the systemic claim frequency index over consecutive historical exposure periods:")
            # Trend Graph 1
            years = ["2021", "2022", "2023", "2024", "2025"]
            freq_index = [0.082, 0.079, 0.071, 0.064, 0.058]
            fig_trend1 = px.line(x=years, y=freq_index, labels={"x": "Financial Year", "y": "Claim Frequency Index"})
            fig_trend1.update_traces(line_color="#1E3A8A", line_width=3, mode="lines+markers")
            fig_trend1.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=250)
            st.plotly_chart(fig_trend1, use_container_width=True)

    with col_e2:
        with st.container(border=True):
            st.markdown("#### **📉 Demographic Cost Projections by Exposure Band**")
            st.write("The average empirical cost trajectory when plotted against policyholder demographic age spectrum boundaries:")
            # Trend Graph 2
            age_bands = ["17-21", "22-25", "26-35", "36-50", "51-65", "66+"]
            mean_costs = [295.40, 210.10, 115.50, 68.20, 52.40, 78.90]
            fig_trend2 = px.line(x=age_bands, y=mean_costs, labels={"x": "Driver Age Group", "y": "Empirical Loss Cost ($)"})
            fig_trend2.update_traces(line_color="#EF4444", line_width=3, mode="lines+markers")
            fig_trend2.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=250)
            st.plotly_chart(fig_trend2, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### **📊 Zero-Inflation Claim Metric Split**")
        st.write("Over **98.2%** of policies generate zero claims. Our modeling architecture leverages a Poisson link function explicitly to handle this skew cleanly.")
        labels = ['No Claims ($0)', 'Minor Claims (<$500)', 'Severe Claims (>$500)']
        values = [98.2, 1.4, 0.4]
        fig_pie = px.pie(names=labels, values=values, color_discrete_sequence=['#1E3A8A', '#3B82F6', '#EF4444'], hole=0.4)
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220)
        st.plotly_chart(fig_pie, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 3: MODEL CHAMPIONS (UPDATED FROM BAR CHART TO LINE GRAPH)
# ────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown("### **Model Tournament Comparisons**")
    
    col_t1, col_t2 = st.columns([1.1, 0.9])
    
    with col_t1:
        with st.container(border=True):
            st.markdown("#### **🏆 Model Performance Matrix**")
            comparison_data = pd.DataFrame({
                "Model Architecture": ["Baseline (Flat Mean)", "Lasso GLM", "Ridge GLM", "Poisson GLM (Champion)"],
                "Cross-Validated MAE": [65.0000, 61.2291, 61.3491, 58.4110],
                "Out-of-Sample Deviance": [2.4419, 2.1102, 2.1149, 1.8492],
                "Gini Coefficient": [0.0000, 0.1849, 0.1812, 0.2491]
            })
            
            formatted_df = comparison_data.copy()
            formatted_df["Cross-Validated MAE"] = formatted_df["Cross-Validated MAE"].map("${:,.2f}".format)
            formatted_df["Out-of-Sample Deviance"] = formatted_df["Out-of-Sample Deviance"].map("{:.4f}".format)
            formatted_df["Gini Coefficient"] = formatted_df["Gini Coefficient"].map("{:.4f}".format)
            st.dataframe(formatted_df, use_container_width=True, hide_index=True)

    with col_t2:
        with st.container(border=True):
            st.markdown("#### **📈 Gini Lift Curve (Line Graph)**")
            st.write("Tracking model performance lift across our tournament configurations:")
            
            # UPDATED: Replaced bar graph with an interactive line graph tracking tournament performance growth
            fig_gini_line = px.line(
                comparison_data, 
                x="Model Architecture", 
                y="Gini Coefficient",
                labels={"Model Architecture": "Model Iteration", "Gini Coefficient": "Gini Score"},
                markers=True
            )
            fig_gini_line.update_traces(line_color="#1E3A8A", line_width=3, marker=dict(size=10, color="#EF4444"))
            fig_gini_line.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=230)
            st.plotly_chart(fig_gini_line, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Underwriting Calculator (Aligned to Notebook Row 85 Pipeline)
# ════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.caption("Adjust policyholder metrics on the fly. Calculations execute directly inside the loaded serialized pipeline script.")

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
            
            # 1. Inputs Section (Aligned to Row 85 Defaults)
            with st.container(border=True):
                st.markdown("#### **Core Underwriting Metrics**")
                col_u1, col_u2, col_u3 = st.columns(3)
                
                with col_u1:
                    st.markdown("**👤 Demographics**")
                    driver_age = st.slider("Driver Age (Age)", 17, 95, 55, key="calc_age")
                    customer_years = st.slider("Insurer Tenure (Customer_years)", 0.0, 25.0, 1.0, step=0.5, key="calc_ten")
                    
                with col_u2:
                    st.markdown("**🚗 Vehicle Parameters**")
                    power = st.number_input("Engine Power (Power)", min_value=10, max_value=800, value=86, key="calc_pow")
                    vehicle_age = st.slider("Vehicle Age (Vehicle_age)", 0, 30, 23, key="calc_v_age")
                    
                with col_u3:
                    st.markdown("**📄 Operational Classification**")
                    risk_type_label = st.segmented_control(
                        "Risk Classification", 
                        options=["Private Passenger", "Motorcycle", "Commercial", "Fleet"], 
                        default="Private Passenger",
                        key="calc_risk_label"
                    )
                    risk_map = {"Motorcycle": "1", "Private Passenger": "2", "Commercial": "3", "Fleet": "4"}
                    risk_type = risk_map.get(risk_type_label, "2")

                with st.expander("⚙️ Secondary Risk Factors (Optional Adjustment)", expanded=False):
                    col_ex1, col_ex2, col_ex3 = st.columns(3)
                    with col_ex1:
                        licence_years = st.slider("Years Licensed", 0, 75, 31, key="calc_lic")
                        policies_in_force = st.number_input("Policies in Force", 1, 10, 1, key="calc_pif")
                        second_driver = st.pills("Second Driver Registry", ["0", "1"], default="0", key="calc_sec_lbl")
                    with col_ex2:
                        fuel_type = st.pills("Fuel Type", ["P", "D", "Unknown"], default="D", key="calc_fuel")
                        doors = st.pills("Door Count", ["4", "2", "3", "5"], default="5", key="calc_doors")
                        area = st.selectbox("Area Code", ["0", "1", "2", "3", "4", "5"], index=0, key="calc_area")
                    with col_ex3:
                        cylinder = st.number_input("Cylinder Capacity (cc)", min_value=100, max_value=8000, value=2953, key="calc_cyl")
                        length = st.number_input("Vehicle Length (meters)", min_value=1.0, max_value=8.0, value=4.61, step=0.01, key="calc_len")
                        weight = st.number_input("Vehicle Weight (kg)", min_value=300, max_value=4000, value=1820, key="calc_weight")
                        
                        # System variables matching Row 85
                        channel = 0
                        payment = 0

            # 2. Reconstruct DataFrame matching the EXACT pandas types your pipeline was fitted on
            input_row = pd.DataFrame([{
                # NUMERIC variables
                "Age": float(driver_age),
                "Licence_years": float(licence_years),
                "Vehicle_age": float(vehicle_age),
                "Customer_years": float(customer_years),
                "Power": float(power),
                "Cylinder_capacity": float(cylinder),
                "Length": float(length),
                "Weight": float(weight),
                "Policies_in_force": float(policies_in_force),
                "Length_missing": float(0.0),
                
                # CATEGORICAL variables (Casted to integers to prevent OneHotEncoder failures)
                "Type_risk": int(risk_type),
                "Type_fuel": str(fuel_type),
                "Area": int(area),
                "Distribution_channel": int(channel),
                "Payment": int(payment),
                "Second_driver": int(second_driver),
                "N_doors": int(doors)
            }])

            # Strict feature matrix arrangement matching training pipeline
            ordered_cols = [
                "Age", "Licence_years", "Vehicle_age", "Customer_years", "Power", 
                "Cylinder_capacity", "Length", "Weight", "Policies_in_force", "Length_missing",
                "Type_risk", "Type_fuel", "Area", "Distribution_channel", "Payment", "Second_driver", "N_doors"
            ]
            input_row = input_row[ordered_cols]

            # 3. Developer Lineage Expander (Collapsible)
            with st.expander("🔍 Developer Lineage Audit (Feature Vector)", expanded=False):
                st.write("This table displays the precise column sequence and data type structures being fed directly into the model:")
                st.dataframe(input_row, use_container_width=True)

            # 4. Process Machine Learning Prediction
            try:
                real_predicted_cost = float(trained_model.predict(input_row)[0])
                st.success(f"🎯 **Authenticated Model Prediction Cost:** `${real_predicted_cost:.2f}`")
                
                # Dynamic mapping custom-tailored to your model's true output window ($220 to $390+)
                if real_predicted_cost <= 222.0: decile = 1
                elif real_predicted_cost <= 230.0: decile = 2
                elif real_predicted_cost <= 242.0: decile = 3
                elif real_predicted_cost <= 255.0: decile = 4
                elif real_predicted_cost <= 270.0: decile = 5
                elif real_predicted_cost <= 290.0: decile = 6
                elif real_predicted_cost <= 315.0: decile = 7
                elif real_predicted_cost <= 340.0: decile = 8
                elif real_predicted_cost <= 375.0: decile = 9
                else: decile = 10

                # 5. Core Mathematical Calculations
                rel_risk = actual_means[decile-1] / portfolio_avg
                safety_loading = 1.30 if driver_age < 25 else 1.15
                recommended_premium = (real_predicted_cost * safety_loading)

                # 6. Commercial Verdict Card Layout
                st.markdown("### **Commercial Underwriting & Referral Verdict**")
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.metric("Risk Placement Band", f"Decile {decile} / 10", f"{rel_risk:.2f}x average risk")
                with col_res2:
                    st.metric("Recommended Premium", f"${recommended_premium:.2f}", f"Includes {int((safety_loading-1)*100)}% Loading Factor")
                with col_res3:
                    if driver_age < 21:
                        status, color, desc = "REFER TO SENIOR CUO", "#EF4444", "Policyholder is under 21. Automatic trigger for manual premium review."
                    elif risk_type == 4: # Int check
                        status, color, desc = "FLEET REVIEW REQUIRED", "#F59E0B", "Commercial fleet classifications require commercial vehicle safety audits."
                    elif decile <= 4:
                        status, color, desc = "AUTO-PASS", "#10B981", "Optimal risk metrics. Fast-track automated rate with no manual intervention."
                    else:
                        status, color, desc = "STANDARD AUDIT", "#F59E0B", "Standard processing. Review claims history before final sign-off."
                        
                    st.markdown(
                        f"""
                        <div class="verdict-card" style="border-top: 4px solid {color}; padding: 12px 15px;">
                            <span style="font-weight: 700; color: {color}; font-size: 0.95rem;">{status}</span><br>
                            <span style="font-size: 0.85rem; color: #475569; line-height: 1.4; display: inline-block; margin-top: 4px;">{desc}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                # 7. Applied Underwriting Rules & Governance Audit
                with st.container(border=True):
                    st.markdown("##### 📝 **Applied Underwriting Rules & Governance Audit**")
                    col_aud1, col_aud2 = st.columns(2)
                    with col_aud1:
                        st.markdown("**Executed Pricing Rules:**")
                        if driver_age < 25:
                            st.caption("⚠️ **High-Risk Demographics:** +30% Risk Loading Applied (Age < 25)")
                        else:
                            st.caption("✅ **Standard Demographics:** +15% Baseline Risk Loading Applied")
                        
                        if power > 150:
                            st.caption("⚠️ **High-Performance Vehicle:** Manual underwriting review recommended (Power > 150 HP)")
                        else:
                            st.caption("✅ **Standard Performance Baseline:** Engine parameters clear automated limits")
                    
                    with col_aud2:
                        st.markdown("**Executed Portfolio Rules:**")
                        if driver_age < 21:
                            st.caption("❌ **Age Restriction Block:** Referral triggered (Driver is under 21)")
                        elif risk_type == 4: # Int check
                            st.caption("📋 **Commercial Protocol:** Fleet risk assessment standards enforced")
                        else:
                            st.caption("✅ **Standard Protocol:** Auto-routing criteria met")

                # 8. Dynamic Premium Deviance Analysis vs Portfolio Baselines
                st.markdown("#### **📊 Live Premium Deviance Analysis vs Portfolio Baselines**")
                model_base_target = 58.4110
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=["Portfolio Avg MAE", "Model Baseline MAE", "Applicant Predicted Cost"],
                    y=[portfolio_avg, model_base_target, real_predicted_cost],
                    mode="lines+markers",
                    line=dict(color="#1E3A8A", width=3, dash="dash"),
                    marker=dict(size=12, color=["#94A3B8", "#3B82F6", "#EF4444"])
                ))
                fig_comp.update_layout(
                    title="<b>Applicant Cost Variance Position Matrix</b>",
                    yaxis=dict(title="Mathematical Valuation Score ($)"),
                    height=240,
                    margin=dict(t=30, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_comp, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Pipeline Matrix execution error: `{str(e)}`")

        prediction_fragment()
