"""Streamlit UI for the Customer Churn Predictor."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import preprocess

# ── paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURES_PATH = BASE_DIR / "feature_names.pkl"
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# ── page config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")

# ── load model ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, feature_names


model, feature_names = load_model()

if model is None:
    st.error("⚠️ Model not found! Please run `python train.py` first.")
    st.stop()

# ── helpers ──────────────────────────────────────────────────────────────
# Build label-encoder mappings from the training data so that the UI can
# translate human-readable selections into the encoded integers the model
# expects.  We cache the result so the CSV is only read once.

@st.cache_resource
def get_label_mappings():
    """Return {column_name: {label_string: encoded_int}} for every
    object column that was label-encoded during preprocessing."""
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.drop(columns=["customerID"], inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    mappings: dict[str, dict[str, int]] = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        le.fit(df[col])
        mappings[col] = {label: int(idx) for idx, label in enumerate(le.classes_)}
    return mappings


label_maps = get_label_mappings()

# ── tabs ─────────────────────────────────────────────────────────────────
tab_predict, tab_insights = st.tabs(["🔮 Predict", "🌍 Global Insights"])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — Single-customer prediction
# ═══════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("Enter customer details")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 50.0, step=0.5)
        total_charges = tenure * monthly_charges
        st.metric("Total Charges (auto)", f"${total_charges:,.2f}")

        contract_label = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_label = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:
        tech_support_label = st.selectbox("Tech Support", ["Yes", "No"])
        online_security_label = st.selectbox("Online Security", ["Yes", "No"])
        paperless_label = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_label = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    if st.button("🚀 Predict", use_container_width=True):
        # Build a single-row dataframe with ALL feature columns set to 0,
        # then fill in the ones the user controls.
        row = {feat: 0 for feat in feature_names}

        # Numeric features
        row["tenure"] = tenure
        row["MonthlyCharges"] = monthly_charges
        row["TotalCharges"] = total_charges

        # Categorical features — encode using the same label mapping
        row["Contract"] = label_maps["Contract"][contract_label]
        row["InternetService"] = label_maps["InternetService"][internet_label]
        row["TechSupport"] = label_maps["TechSupport"][tech_support_label]
        row["OnlineSecurity"] = label_maps["OnlineSecurity"][online_security_label]
        row["PaperlessBilling"] = label_maps["PaperlessBilling"][paperless_label]
        row["PaymentMethod"] = label_maps["PaymentMethod"][payment_label]

        input_df = pd.DataFrame([row], columns=feature_names)

        # Predict
        proba = model.predict_proba(input_df)[0, 1]
        pct = proba * 100

        st.markdown("---")
        st.metric("Churn Probability", f"{pct:.1f}%")

        if pct > 50:
            st.error("🔴 **High Risk** — This customer is likely to churn.")
        else:
            st.success("🟢 **Low Risk** — This customer is likely to stay.")

        # SHAP waterfall for this individual prediction
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(input_df)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_vals[0], max_display=12, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Red bars push toward churn, blue bars push away.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — Global Insights (pre-generated SHAP plots)
# ═══════════════════════════════════════════════════════════════════════
with tab_insights:
    st.subheader("Global Feature Importance")

    bar_path = OUTPUT_DIR / "shap_bar.png"
    beeswarm_path = OUTPUT_DIR / "shap_beeswarm.png"

    if not bar_path.exists() or not beeswarm_path.exists():
        st.warning("SHAP plots not found. Please run `python explain.py` first.")
    else:
        st.image(str(bar_path), caption="Mean |SHAP| — Global Feature Importance")
        st.markdown(
            """
            **What this shows:** Each bar represents the average absolute SHAP value
            for a feature across all test-set predictions. Longer bars indicate
            features that have a greater overall impact on the model's churn
            predictions.
            """
        )

        st.markdown("---")

        st.image(str(beeswarm_path), caption="SHAP Beeswarm — Direction of Impact")
        st.markdown(
            """
            **What this shows:** Each dot is one customer. The colour indicates the
            feature value (red = high, blue = low). Dots pushed to the right increase
            the predicted churn probability; dots pushed to the left decrease it.
            This reveals *how* each feature drives predictions, not just *how much*.
            """
        )
