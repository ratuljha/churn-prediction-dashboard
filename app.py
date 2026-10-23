"""
app.py
Interactive Customer Churn Analytics & Risk Dashboard
Built with Streamlit, Plotly, and XGBoost.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction & Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# DATA & MODEL LOADING (CACHED)
# -------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "churn_model.pkl")
    features_path = os.path.join(base_dir, "model", "feature_names.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        st.error("Model artifacts not found! Ensure model/churn_model.pkl exists.")
        return None, None
        
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    return model, feature_names

@st.cache_data
def load_historical_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not os.path.exists(data_path):
        return pd.DataFrame()
    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df

model, feature_names = load_model_artifacts()
df_raw = load_historical_data()

# -------------------------------------------------------------
# HEADER SECTION
# -------------------------------------------------------------
st.title("📊 Customer Churn Intelligence & Risk Dashboard")
st.markdown(
    "Real-time customer attrition forecasting powered by **XGBoost** and explainable data analytics. "
    "Designed for retention managers to proactively identify at-risk subscribers."
)

# -------------------------------------------------------------
# TABS
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🎯 Individual Customer Risk Predictor", 
    "📈 Cohort & Segment Analytics", 
    "📁 Dataset Explorer"
])

# =============================================================
# TAB 1: INDIVIDUAL PREDICTOR
# =============================================================
with tab1:
    st.subheader("Individual Customer Profile & Inference")
    st.markdown("Adjust subscriber parameters below to compute live churn probability.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account & Tenure**")
        tenure = st.slider("Tenure (Months with company)", min_value=1, max_value=72, value=12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=75.0, step=0.5)
        total_charges = tenure * monthly_charges
        st.caption(f"Estimated Total Charges: **${total_charges:,.2f}**")

    with col2:
        st.markdown("**Services Subscribed**")
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("**Billing & Demographics**")
        payment_method = st.selectbox(
            "Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    # Predict Button
    if st.button("🚀 Calculate Churn Probability", type="primary", use_container_width=True):
        if model is not None and feature_names is not None:
            # Build input vector
            input_dict = {f: 0 for f in feature_names}
            
            # Numeric
            input_dict["SeniorCitizen"] = 1 if senior_citizen == "Yes" else 0
            input_dict["tenure"] = tenure
            input_dict["MonthlyCharges"] = monthly_charges
            input_dict["TotalCharges"] = total_charges

            # Categorical dummies
            if partner == "Yes": input_dict["Partner_Yes"] = 1
            if dependents == "Yes": input_dict["Dependents_Yes"] = 1
            if internet_service == "Fiber optic": input_dict["InternetService_Fiber optic"] = 1
            elif internet_service == "No": input_dict["InternetService_No"] = 1

            if online_security == "Yes": input_dict["OnlineSecurity_Yes"] = 1
            elif online_security == "No internet service": input_dict["OnlineSecurity_No internet service"] = 1

            if tech_support == "Yes": input_dict["TechSupport_Yes"] = 1
            elif tech_support == "No internet service": input_dict["TechSupport_No internet service"] = 1

            if device_protection == "Yes": input_dict["DeviceProtection_Yes"] = 1
            elif device_protection == "No internet service": input_dict["DeviceProtection_No internet service"] = 1

            if contract == "One year": input_dict["Contract_One year"] = 1
            elif contract == "Two year": input_dict["Contract_Two year"] = 1

            if paperless_billing == "Yes": input_dict["PaperlessBilling_Yes"] = 1

            if payment_method == "Electronic check": input_dict["PaymentMethod_Electronic check"] = 1
            elif payment_method == "Mailed check": input_dict["PaymentMethod_Mailed check"] = 1
            elif payment_method == "Credit card (automatic)": input_dict["PaymentMethod_Credit card (automatic)"] = 1

            # Create DataFrame with exact feature order
            X_input = pd.DataFrame([input_dict])[feature_names]
            churn_prob = model.predict_proba(X_input)[0, 1]
            churn_pct = churn_prob * 100

            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 1.5])

            with res_col1:
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=churn_pct,
                    number={'suffix': "%", 'font': {'size': 36}},
                    title={'text': "Predicted Churn Probability", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#EF4444" if churn_pct >= 60 else ("#F59E0B" if churn_pct >= 35 else "#10B981")},
                        'steps': [
                            {'range': [0, 35], 'color': "#ECFDF5"},
                            {'range': [35, 60], 'color': "#FEF3C7"},
                            {'range': [60, 100], 'color': "#FEE2E2"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 3},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_col2:
                if churn_pct >= 60:
                    st.error(f"⚠️ **HIGH RISK SUBSCRIBER ({churn_pct:.1f}%)**")
                    st.markdown("""
                    **Retention Recommendations:**
                    - **Urgent Outreach:** Customer is on a short-term commitment with high risk drivers.
                    - **Action 1:** Offer a 1-year contract discount voucher (-15% billing discount).
                    - **Action 2:** Proactively bundle complimentary Tech Support to enhance stickiness.
                    - **Action 3:** Transition customer from Electronic Check to automated credit card payment.
                    """)
                elif churn_pct >= 35:
                    st.warning(f"⚡ **MODERATE RISK SUBSCRIBER ({churn_pct:.1f}%)**")
                    st.markdown("""
                    **Retention Recommendations:**
                    - **Engagement Check:** Check satisfaction scores regarding broadband stability.
                    - **Action:** Propose value-added bundle (Device Protection or streaming tier).
                    """)
                else:
                    st.success(f"✅ **LOW RISK / LOYAL SUBSCRIBER ({churn_pct:.1f}%)**")
                    st.markdown("""
                    **Retention Status:**
                    - Customer shows high retention markers (extended tenure and stable payment methods).
                    - **Action:** Eligible for loyalty upgrade campaigns and premium add-ons.
                    """)

# =============================================================
# TAB 2: COHORT ANALYTICS
# =============================================================
with tab2:
    st.subheader("Subscriber Population Insights & Risk Cohorts")
    if not df_raw.empty:
        c1, c2 = st.columns(2)

        with c1:
            # Contract vs Churn
            fig_contract = px.histogram(
                df_raw, 
                x="Contract", 
                color="Churn", 
                barmode="group",
                color_discrete_map={"Yes": "#EF4444", "No": "#3B82F6"},
                title="<b>Churn Distribution by Contract Type</b>"
            )
            fig_contract.update_layout(height=350)
            st.plotly_chart(fig_contract, use_container_width=True)

        with c2:
            # Internet Service vs Churn
            fig_internet = px.histogram(
                df_raw, 
                x="InternetService", 
                color="Churn", 
                barmode="group",
                color_discrete_map={"Yes": "#EF4444", "No": "#3B82F6"},
                title="<b>Churn Distribution by Internet Service</b>"
            )
            fig_internet.update_layout(height=350)
            st.plotly_chart(fig_internet, use_container_width=True)

        # Monthly charges distribution
        fig_charges = px.box(
            df_raw, 
            x="Contract", 
            y="MonthlyCharges", 
            color="Churn",
            color_discrete_map={"Yes": "#EF4444", "No": "#3B82F6"},
            title="<b>Monthly Billing Distribution vs Churn Status</b>"
        )
        fig_charges.update_layout(height=380)
        st.plotly_chart(fig_charges, use_container_width=True)

# =============================================================
# TAB 3: DATASET EXPLORER
# =============================================================
with tab3:
    st.subheader("Subscriber Database Explorer")
    if not df_raw.empty:
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            contract_filter = st.multiselect("Filter Contract", options=df_raw["Contract"].unique(), default=df_raw["Contract"].unique())
        with filter_col2:
            churn_filter = st.multiselect("Filter Churn Status", options=df_raw["Churn"].unique(), default=df_raw["Churn"].unique())
        with filter_col3:
            internet_filter = st.multiselect("Filter Internet Service", options=df_raw["InternetService"].unique(), default=df_raw["InternetService"].unique())

        df_filtered = df_raw[
            (df_raw["Contract"].isin(contract_filter)) &
            (df_raw["Churn"].isin(churn_filter)) &
            (df_raw["InternetService"].isin(internet_filter))
        ]

        st.dataframe(df_filtered.head(100), use_container_width=True)
        st.caption(f"Showing {min(100, len(df_filtered))} of {len(df_filtered):,} matching records.")

        # Download CSV
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Segment as CSV",
            data=csv,
            file_name="churn_filtered_segment.csv",
            mime="text/csv",
            use_container_width=True
        )
