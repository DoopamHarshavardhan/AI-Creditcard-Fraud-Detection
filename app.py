import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import shap

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Fraud Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM DARK UI
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

div[data-testid="stSidebar"] {
    background-color: #111827;
}

.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
}

h1, h2, h3, h4 {
    color: #F9FAFB;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# SHAP EXPLAINER
explainer = shap.TreeExplainer(model)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("💳 Fraud Detection System")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Upload & Predict",
        "Single Transaction Prediction",
        "Model Explainability",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
### Model Information

- Algorithm: XGBoost
- SMOTE Balanced
- ROC-AUC: 0.97
- Recall: 85%
- Dataset: Kaggle Credit Card Fraud
""")

# =========================================================
# TITLE
# =========================================================

st.title("💳 AI-Powered Credit Card Fraud Detection System")

st.markdown("""
Machine Learning based fraud detection system using:

- XGBoost
- SMOTE
- Isolation Forest
- Local Outlier Factor
- Explainable AI (SHAP)
""")

# =========================================================
# FEATURE NAMES
# =========================================================

feature_names = [
    'Time','V1','V2','V3','V4','V5','V6','V7','V8',
    'V9','V10','V11','V12','V13','V14','V15',
    'V16','V17','V18','V19','V20','V21','V22',
    'V23','V24','V25','V26','V27','V28','Amount'
]

# =========================================================
# DASHBOARD
# =========================================================

if menu == "Dashboard":

    st.header("📊 Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("ROC-AUC Score", "0.97")
    col2.metric("Fraud Recall", "85%")
    col3.metric("Model Accuracy", "99.86%")

    st.markdown("---")

    st.subheader("📌 Fraud Detection Pipeline")

    st.write("""
    1. Data Preprocessing  
    2. Feature Scaling  
    3. SMOTE Balancing  
    4. XGBoost Classification  
    5. Isolation Forest Detection  
    6. Local Outlier Factor Detection  
    7. Explainable AI Analysis  
    """)

    st.markdown("---")

    st.subheader("📈 Model Comparison")

    comparison = pd.DataFrame({

        'Model': [
            'Logistic Regression',
            'Isolation Forest',
            'Local Outlier Factor',
            'XGBoost + SMOTE'
        ],

        'Recall': [
            0.63,
            0.35,
            0.02,
            0.85
        ]
    })

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(comparison['Model'], comparison['Recall'])

    plt.xticks(rotation=10)

    plt.ylabel("Recall")

    st.pyplot(fig)

    # Feature Importance
    st.subheader("🔥 Top Fraud Indicators")

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })

    importance_df = importance_df.sort_values(
        by='Importance',
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index('Feature')
    )

# =========================================================
# UPLOAD & PREDICT
# =========================================================

elif menu == "Upload & Predict":

    st.header("📂 Upload Transaction Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Dataset")

        st.dataframe(data.head())

        # Remove target column
        if 'Class' in data.columns:
            features = data.drop('Class', axis=1)
        else:
            features = data

        # Scale
        scaled_data = scaler.transform(features)

        # Predict with loading spinner
        with st.spinner("Analyzing Transactions..."):

            predictions = model.predict(scaled_data)

            probabilities = model.predict_proba(scaled_data)[:,1]

        # Add predictions
        data['Prediction'] = predictions
        data['Fraud_Probability'] = probabilities

        # Risk levels
        def risk_level(prob):

            if prob > 0.8:
                return "High Risk"

            elif prob > 0.4:
                return "Medium Risk"

            else:
                return "Low Risk"

        data['Risk_Level'] = data['Fraud_Probability'].apply(risk_level)

        # Metrics
        fraud_count = (data['Prediction'] == 1).sum()

        normal_count = (data['Prediction'] == 0).sum()

        st.subheader("📊 Detection Summary")

        col1, col2 = st.columns(2)

        col1.metric("Fraud Transactions", fraud_count)
        col2.metric("Normal Transactions", normal_count)

        # Pie chart
        st.subheader("📈 Fraud Distribution")

        fig, ax = plt.subplots()

        ax.pie(
            [fraud_count, normal_count],
            labels=['Fraud', 'Normal'],
            autopct='%1.1f%%'
        )

        st.pyplot(fig)

        # Probability distribution
        st.subheader("📊 Fraud Probability Distribution")

        fig, ax = plt.subplots()

        ax.hist(
            data['Fraud_Probability'],
            bins=30
        )

        st.pyplot(fig)

        # Correlation Heatmap
        st.subheader("📌 Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(12,8))

        sns.heatmap(
            data.corr(numeric_only=True),
            cmap='coolwarm'
        )

        st.pyplot(fig)

        # Top suspicious transactions
        st.subheader("🚨 Top Suspicious Transactions")

        suspicious = data.sort_values(
            by='Fraud_Probability',
            ascending=False
        ).head(10)

        st.dataframe(suspicious)

        # Full results
        st.subheader("📋 Full Prediction Results")

        st.dataframe(data)

        # Download report
        csv = data.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="⬇ Download Prediction Report",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    else:
        st.info("Upload a CSV file to begin.")

# =========================================================
# SINGLE TRANSACTION PREDICTION
# =========================================================

elif menu == "Single Transaction Prediction":

    st.header("💳 Single Transaction Fraud Prediction")

    st.write("Enter transaction details below.")

    user_input = []

    cols = st.columns(3)

    for i, feature in enumerate(feature_names):

        value = cols[i % 3].number_input(
            feature,
            value=0.0
        )

        user_input.append(value)

    # Random transaction simulator
    if st.button("Generate Random Transaction"):

        random_tx = np.random.normal(size=(1,30))

        pred = model.predict(random_tx)[0]

        prob = model.predict_proba(random_tx)[0][1]

        st.write("Prediction:", pred)

        st.write(f"Fraud Probability: {prob:.2%}")

    # Prediction
    if st.button("Predict Transaction"):

        input_array = np.array(user_input).reshape(1, -1)

        scaled_input = scaler.transform(input_array)

        prediction = model.predict(scaled_input)[0]

        probability = model.predict_proba(
            scaled_input
        )[0][1]

        st.subheader("Prediction Result")

        # Fraud alert
        if prediction == 1:

            st.error("🚨 HIGH RISK FRAUD DETECTED")

            st.balloons()

            st.warning("""
            Immediate review recommended.
            Transaction exceeds fraud threshold.
            """)

        else:

            st.success("✅ Legitimate Transaction")

        # Gauge chart
        st.subheader("📊 Fraud Risk Score")

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,

            title = {'text': "Fraud Probability"},

            gauge = {
                'axis': {'range': [0,100]},
                'bar': {'color': "red"},
            }
        ))

        st.plotly_chart(fig)

        st.write(f"Fraud Probability: {probability:.2%}")

        # Risk level
        if probability > 0.8:
            st.error("High Risk")

        elif probability > 0.4:
            st.warning("Medium Risk")

        else:
            st.success("Low Risk")

        # SHAP Explainability
        st.subheader("🔍 Explainable AI Analysis")

        shap_values = explainer.shap_values(scaled_input)

        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Impact': shap_values[0]
        })

        shap_df = shap_df.sort_values(
            by='Impact',
            ascending=False
        )

        st.dataframe(shap_df.head(10))

# =========================================================
# MODEL EXPLAINABILITY
# =========================================================

elif menu == "Model Explainability":

    st.header("🧠 Explainable AI")

    st.write("""
    SHAP (SHapley Additive exPlanations) is used to explain
    how each feature contributes to fraud prediction.
    """)

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })

    importance_df = importance_df.sort_values(
        by='Importance',
        ascending=False
    )

    st.subheader("🔥 Global Feature Importance")

    st.dataframe(importance_df)

    fig, ax = plt.subplots(figsize=(10,6))

    ax.barh(
        importance_df['Feature'][:10],
        importance_df['Importance'][:10]
    )

    plt.xlabel("Importance")

    st.pyplot(fig)

# =========================================================
# ABOUT PROJECT
# =========================================================

elif menu == "About Project":

    st.header("📘 About This Project")

    st.write("""
    This project focuses on detecting fraudulent credit card
    transactions using Machine Learning and anomaly detection.

    ### Technologies Used
    - Python
    - Pandas
    - Scikit-Learn
    - XGBoost
    - Streamlit
    - Plotly
    - SHAP

    ### Machine Learning Techniques
    - SMOTE Balancing
    - XGBoost Classification
    - Isolation Forest
    - Local Outlier Factor
    - Explainable AI

    ### Features
    - Fraud Prediction
    - Risk Scoring
    - Analytics Dashboard
    - CSV Upload Detection
    - Real-Time Prediction
    - Explainable AI
    - Downloadable Reports

    ### Final Model Performance
    - Accuracy: 99.86%
    - Recall: 85%
    - ROC-AUC: 0.97
    """)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<center>
AI-Powered Credit Card Fraud Detection System • AIML Internship Project
</center>
""", unsafe_allow_html=True)