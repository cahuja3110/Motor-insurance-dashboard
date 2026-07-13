"""
Applied Machine Learning — Week 1: Core ML
Interactive Streamlit companion to class1_code.ipynb

Run with:  streamlit run class1_app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn import set_config
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

set_config(transform_output="pandas")

# ────────────────────────────────────────────────────────────────────────────
# Page config + theming
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Core ML — Week 1",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Open Sans + subtle visual polish
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
      html, body, [class*="css"]  { font-family: 'Open Sans', sans-serif; }
      .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
      [data-testid="stMetric"] {
          background: rgba(120, 120, 200, 0.06);
          border: 1px solid rgba(120, 120, 200, 0.15);
          padding: 14px 18px; border-radius: 12px;
      }
      [data-testid="stMetricLabel"] { font-weight: 600; }
      .hero {
          background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
          color: white; padding: 28px 32px; border-radius: 16px;
          margin-bottom: 20px;
      }
      .hero h1 { color: white; margin: 0; font-weight: 700; }
      .hero p  { color: rgba(255,255,255,0.85); margin: 6px 0 0 0; font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

RANDOM_STATE = 42

MODEL_FACTORIES = {
    "Baseline (majority)":  lambda: DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
    "Logistic Regression":  lambda: LogisticRegression(max_iter=1000, class_weight="balanced",
                                                       random_state=RANDOM_STATE),
    "Decision Tree":        lambda: DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                                           random_state=RANDOM_STATE),
    "Random Forest":        lambda: RandomForestClassifier(n_estimators=300, max_depth=10,
                                                           class_weight="balanced", n_jobs=-1,
                                                           random_state=RANDOM_STATE),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(max_iter=300,
                                                                   random_state=RANDOM_STATE),
}


# ────────────────────────────────────────────────────────────────────────────
# Cached data + model functions
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str = "class1_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns="customerID")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


@st.cache_data
def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns="Churn")
    y = df["Churn"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(),
                make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_include=object)),
    ])


@st.cache_resource(show_spinner=False)
def train_pipeline(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe = Pipeline([("prep", make_preprocessor()),
                     ("model", MODEL_FACTORIES[model_name]())])
    pipe.fit(X_train, y_train)
    return pipe


@st.cache_data(show_spinner=False)
def evaluate(model_name: str, _pipe: Pipeline, X_test, y_test, X_train, y_train,
             do_cv: bool = True) -> dict:
    """Compute metrics. _pipe prefix tells Streamlit not to hash it."""
    y_pred = _pipe.predict(X_test)
    try:
        y_proba = _pipe.predict_proba(X_test)[:, 1]
        auc_test = roc_auc_score(y_test, y_proba)
    except (AttributeError, ValueError):
        y_proba = None
        auc_test = np.nan

    cv_auc = np.nan
    if do_cv and not isinstance(_pipe.named_steps["model"], DummyClassifier):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_auc = cross_val_score(_pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1).mean()

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_churn": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_churn": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1_churn": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "auc_test": auc_test,
        "auc_cv": cv_auc,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <h1>🧠 Core Machine Learning</h1>
      <p>Applied ML · Week 1 · Predicting Telco customer churn end to end</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.dialog("About this app")
def show_about():
    st.markdown("""
**What's here**

A live companion to the Week 1 notebook. Each tab maps to a step of the ML workflow:

1. **Explore** — inspect the dataset and look for patterns
2. **Preprocess** — see how the pipeline transforms raw columns
3. **Train & compare** — train every model in one click and rank them
4. **Inspect** — drill into one model: confusion matrix, ROC, importances
5. **Predict** — build a customer profile and watch the model react

**Stack**: scikit-learn, plotly, Streamlit. All caching uses `@st.cache_data` and `@st.cache_resource`.
    """)


col1, col2 = st.columns([5, 1])
with col2:
    if st.button("ℹ️ About", use_container_width=True):
        show_about()


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Setup")
    uploaded = st.file_uploader("Upload `class1_dataset.csv`", type="csv",
                                 help="Or place the file next to this script.")

    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05,
                          help="Stratified by Churn.")

    st.divider()
    st.caption("Built for Applied ML · Bayes Business School")


try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df = df.drop(columns="customerID", errors="ignore")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
        df["Churn"] = (df["Churn"] == "Yes").astype(int) if df["Churn"].dtype == object else df["Churn"]
    else:
        df = load_data()
except FileNotFoundError:
    st.error("Couldn't find `class1_dataset.csv`. Upload it from the sidebar or place it next to this file.")
    st.stop()

X_train, X_test, y_train, y_test = split_data(df, test_size=test_size, random_state=RANDOM_STATE)


# ────────────────────────────────────────────────────────────────────────────
# Top-level metrics
# ────────────────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers", f"{len(df):,}")
m2.metric("Features", f"{df.shape[1] - 1}")
m3.metric("Churn rate", f"{df['Churn'].mean():.1%}")
m4.metric("Train / test split", f"{len(X_train):,} / {len(X_test):,}")


# ────────────────────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────────────────────
tab_explore, tab_prep, tab_train, tab_inspect, tab_predict = st.tabs(
    ["📊 Explore", "🧰 Preprocess", "🏋️ Train & compare", "🔬 Inspect", "🎯 Predict"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Explore
# ════════════════════════════════════════════════════════════════════════════
with tab_explore:
    view = st.segmented_control(
        "View",
        options=["Data", "Categorical churn drivers", "Numeric distributions", "Correlation"],
        default="Data",
    )

    if view == "Data":
        st.dataframe(
            df,
            use_container_width=True,
            height=420,
            column_config={
                "tenure":         st.column_config.ProgressColumn("Tenure (months)", min_value=0, max_value=int(df["tenure"].max()), format="%d"),
                "MonthlyCharges": st.column_config.NumberColumn("Monthly Charges", format="€%.2f"),
                "TotalCharges":   st.column_config.NumberColumn("Total Charges",   format="€%.2f"),
                "Churn":          st.column_config.NumberColumn("Churn", format="%d"),
            },
        )

    elif view == "Categorical churn drivers":
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        selected = st.pills("Feature", options=cat_cols,
                            default="Contract" if "Contract" in cat_cols else cat_cols[0])
        if selected:
            tmp = df.assign(Churn=df["Churn"].map({0: "Stayed", 1: "Churned"}))
            fig = px.histogram(tmp, x=selected, color="Churn", barmode="group",
                               text_auto=True, color_discrete_sequence=["#10b981", "#ef4444"])
            fig.update_layout(height=480, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

            churn_rate = (df.groupby(selected)["Churn"].mean()
                          .sort_values(ascending=False).to_frame("Churn rate"))
            st.dataframe(
                churn_rate,
                column_config={"Churn rate": st.column_config.ProgressColumn(
                    "Churn rate", format="%.1f%%", min_value=0, max_value=1)},
                use_container_width=True,
            )

    elif view == "Numeric distributions":
        num_cols = df.select_dtypes(include=np.number).columns.drop("Churn").tolist()
        selected = st.pills("Feature", options=num_cols, default="tenure")
        if selected:
            tmp = df.assign(Churn=df["Churn"].map({0: "Stayed", 1: "Churned"}))
            fig = px.histogram(tmp, x=selected, color="Churn", marginal="box", nbins=40,
                               color_discrete_sequence=["#10b981", "#ef4444"], opacity=0.75)
            fig.update_layout(height=480, barmode="overlay", margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    elif view == "Correlation":
        num_df = df.select_dtypes(include=np.number)
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlations among numeric features")
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Preprocess
# ════════════════════════════════════════════════════════════════════════════
with tab_prep:
    st.markdown("#### How the pipeline transforms raw columns")
    st.caption("Numeric columns are standardised. Categorical columns are one-hot encoded with `handle_unknown='ignore'`.")

    prep = make_preprocessor().fit(X_train)
    X_train_t = prep.transform(X_train)

    a, b = st.columns(2)
    a.metric("Raw features", X_train.shape[1])
    b.metric("After encoding", X_train_t.shape[1])

    with st.expander("Preview transformed training set", expanded=False):
        st.dataframe(X_train_t.head(20), use_container_width=True, height=320)

    with st.expander("Inspect a specific column", expanded=True):
        col = st.selectbox("Column", X_train.columns.tolist(), index=list(X_train.columns).index("Contract"))
        if X_train[col].dtype == object:
            ohe_cols = [c for c in X_train_t.columns if c.startswith(f"cat__{col}_")]
            left, right = st.columns(2)
            with left:
                st.markdown(f"**Raw `{col}`**")
                st.dataframe(X_train[[col]].head(8).reset_index(drop=True), height=320)
            with right:
                st.markdown(f"**One-hot encoded ({len(ohe_cols)} columns)**")
                st.dataframe(X_train_t[ohe_cols].head(8).reset_index(drop=True), height=320)
        else:
            scaled_col = f"num__{col}"
            left, right = st.columns(2)
            with left:
                st.markdown(f"**Raw `{col}`** — mean {X_train[col].mean():.2f}, sd {X_train[col].std():.2f}")
                st.dataframe(X_train[[col]].head(8).reset_index(drop=True), height=320)
            with right:
                st.markdown(f"**Standardised** — mean {X_train_t[scaled_col].mean():.2f}, sd {X_train_t[scaled_col].std():.2f}")
                st.dataframe(X_train_t[[scaled_col]].head(8).reset_index(drop=True), height=320)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Train & compare
# ════════════════════════════════════════════════════════════════════════════
with tab_train:
    chosen = st.multiselect(
        "Models to train",
        options=list(MODEL_FACTORIES.keys()),
        default=list(MODEL_FACTORIES.keys()),
    )
    do_cv = st.toggle("Include 5-fold cross-validation (slower)", value=True)

    if st.button("🚀 Train and evaluate", type="primary", use_container_width=False):
        results = []
        with st.status("Training models...", expanded=True) as status:
            for name in chosen:
                st.write(f"→ {name}")
                pipe = train_pipeline(name, X_train, y_train)
                metrics = evaluate(name, pipe, X_test, y_test, X_train, y_train, do_cv=do_cv)
                results.append(metrics)
            status.update(label=f"Trained {len(results)} models", state="complete")

        st.session_state["results"] = results
        st.toast(f"Trained {len(results)} models successfully", icon="✅")

    if "results" in st.session_state:
        results = st.session_state["results"]
        res_df = pd.DataFrame([{
            "Model": r["model"],
            "Accuracy": r["accuracy"],
            "Precision (Churn)": r["precision_churn"],
            "Recall (Churn)": r["recall_churn"],
            "F1 (Churn)": r["f1_churn"],
            "ROC-AUC (test)": r["auc_test"],
            "ROC-AUC (CV)": r["auc_cv"],
        } for r in results])

        st.markdown("##### Results")
        st.dataframe(
            res_df.set_index("Model"),
            use_container_width=True,
            column_config={
                col: st.column_config.ProgressColumn(col, format="%.3f", min_value=0, max_value=1)
                for col in res_df.columns if col != "Model"
            },
        )

        long_df = res_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(long_df, x="Model", y="Score", color="Metric", barmode="group",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(yaxis_range=[0, 1], height=440, margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

        if res_df["ROC-AUC (test)"].notna().any():
            best = res_df.loc[res_df["ROC-AUC (test)"].idxmax()]
            st.success(f"**Best by ROC-AUC:** {best['Model']} — {best['ROC-AUC (test)']:.3f}")
    else:
        st.info("Choose models and click **Train and evaluate** to populate this tab.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Inspect
# ════════════════════════════════════════════════════════════════════════════
with tab_inspect:
    pick = st.pills(
        "Model",
        options=[n for n in MODEL_FACTORIES.keys() if n != "Baseline (majority)"],
        default="HistGradientBoosting",
    )
    threshold = st.slider("Decision threshold (probability of churn)", 0.05, 0.95, 0.50, 0.05,
                          help="Lower = catch more churners (higher recall, lower precision).")

    if pick:
        pipe = train_pipeline(pick, X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.3f}")
        m2.metric("Precision (Churn)", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
        m3.metric("Recall (Churn)",    f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
        m4.metric("ROC-AUC",           f"{roc_auc_score(y_test, y_proba):.3f}")

        view2 = st.segmented_control(
            "View", options=["Confusion matrix", "ROC curve", "Feature importance", "Classification report"],
            default="Confusion matrix",
        )

        if view2 == "Confusion matrix":
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                x=["Predicted: No Churn", "Predicted: Churn"],
                y=["Actual: No Churn", "Actual: Churn"],
            )
            fig.update_layout(height=420, margin=dict(t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

            tn, fp, fn, tp = cm.ravel()
            c1, c2 = st.columns(2)
            c1.markdown(f"- **True positives**: {tp} churners correctly flagged\n"
                        f"- **False negatives**: {fn} churners we'd miss (lost revenue)")
            c2.markdown(f"- **True negatives**: {tn} loyal customers correctly left alone\n"
                        f"- **False positives**: {fp} unnecessary retention offers")

        elif view2 == "ROC curve":
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.3f}",
                                     line=dict(color="#4f46e5", width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                                     line=dict(color="#9ca3af", dash="dash")))
            # mark current threshold
            idx = np.argmin(np.abs(thresholds - threshold))
            fig.add_trace(go.Scatter(x=[fpr[idx]], y=[tpr[idx]], mode="markers",
                                     marker=dict(size=14, color="#ec4899"),
                                     name=f"Threshold = {threshold:.2f}"))
            fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                              height=440, margin=dict(t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

        elif view2 == "Feature importance":
            model = pipe.named_steps["model"]
            feat = pipe.named_steps["prep"].get_feature_names_out()
            if hasattr(model, "feature_importances_"):
                imp = pd.Series(model.feature_importances_, index=feat)
                label = "Importance"
            elif hasattr(model, "coef_"):
                imp = pd.Series(model.coef_[0], index=feat)
                label = "Coefficient (positive → pushes toward Churn)"
            else:
                imp = None

            if imp is None:
                st.info("This model doesn't expose feature importances.")
            else:
                top = imp.reindex(imp.abs().sort_values(ascending=False).index).head(15)
                fig = px.bar(x=top.values, y=top.index, orientation="h",
                             labels={"x": label, "y": ""},
                             color=top.values, color_continuous_scale="RdBu_r",
                             color_continuous_midpoint=0)
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(height=540, margin=dict(t=20, b=0), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        elif view2 == "Classification report":
            report = classification_report(y_test, y_pred,
                                           target_names=["No Churn", "Churn"], output_dict=True)
            report_df = pd.DataFrame(report).T
            st.dataframe(
                report_df.round(3), use_container_width=True,
                column_config={
                    "precision": st.column_config.ProgressColumn("Precision", format="%.3f", min_value=0, max_value=1),
                    "recall":    st.column_config.ProgressColumn("Recall",    format="%.3f", min_value=0, max_value=1),
                    "f1-score":  st.column_config.ProgressColumn("F1",        format="%.3f", min_value=0, max_value=1),
                    "support":   st.column_config.NumberColumn("Support", format="%d"),
                },
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Predict
# ════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.caption("Build a customer profile, then watch the model react in real time.")

    pick_pred = st.pills(
        "Model",
        options=[n for n in MODEL_FACTORIES.keys() if n != "Baseline (majority)"],
        default="HistGradientBoosting",
        key="pred_model",
    )

    @st.fragment
    def prediction_fragment():
        """Re-runs independently of the rest of the page when inputs change."""
        if not pick_pred:
            st.info("Pick a model above.")
            return

        pipe = train_pipeline(pick_pred, X_train, y_train)

        st.markdown("##### Customer profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.slider("Monthly charges (€)", 18.0, 120.0, 65.0, 0.5)
            total = st.number_input("Total charges (€)", min_value=0.0, value=float(tenure * monthly), step=10.0)
        with c2:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            payment  = st.selectbox("Payment method",
                                    ["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)", "Credit card (automatic)"])
        with c3:
            gender   = st.selectbox("Gender", ["Female", "Male"])
            senior   = st.selectbox("Senior citizen", ["No", "Yes"])
            partner  = st.selectbox("Partner", ["Yes", "No"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])

        # Defaults for remaining columns
        defaults = X_train.iloc[0].to_dict()
        defaults.update({
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": "No",
            "tenure": tenure, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": internet,
            "OnlineSecurity": "No" if internet != "No" else "No internet service",
            "OnlineBackup":   "No" if internet != "No" else "No internet service",
            "DeviceProtection": "No" if internet != "No" else "No internet service",
            "TechSupport":    "No" if internet != "No" else "No internet service",
            "StreamingTV":    "No" if internet != "No" else "No internet service",
            "StreamingMovies":"No" if internet != "No" else "No internet service",
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": total,
        })
        profile = pd.DataFrame([defaults])[X_train.columns]

        proba = pipe.predict_proba(profile)[0, 1]

        st.divider()
        left, right = st.columns([1, 2])
        with left:
            st.metric(
                "Probability of churn",
                f"{proba:.1%}",
                delta=f"{(proba - df['Churn'].mean()) * 100:+.1f} pp vs base rate",
                delta_color="inverse",
            )
            verdict = "🔴 High churn risk" if proba >= 0.5 else "🟢 Likely to stay"
            st.markdown(f"### {verdict}")

        with right:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 40}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#4f46e5"},
                    "steps": [
                        {"range": [0, 30],  "color": "#dcfce7"},
                        {"range": [30, 60], "color": "#fef3c7"},
                        {"range": [60, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "#ef4444", "width": 3},
                                  "thickness": 0.75, "value": 50},
                },
            ))
            fig.update_layout(height=260, margin=dict(t=10, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

        with st.popover("Show feature vector sent to model"):
            st.dataframe(profile.T.rename(columns={0: "value"}), use_container_width=True, height=400)

    prediction_fragment()

    st.divider()
    st.caption("Was this useful?")
    feedback = st.feedback("thumbs")
    if feedback is not None:
        st.toast("Thanks for the feedback", icon="💌")
