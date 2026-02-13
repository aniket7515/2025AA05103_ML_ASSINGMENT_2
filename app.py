import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# ---------------------------------------
# Load artifacts
# ---------------------------------------

models = joblib.load("model/churn_models.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")
threshold = joblib.load("model/decision_threshold.pkl")

st.set_page_config(page_title="Bank Churn Prediction", layout="centered")
st.title("🏦 Bank Customer Churn Prediction")
st.markdown(
    "Name: Aniket Dhokane"
    "BITS ID: 2025AA05103"
)
st.markdown(
    "Upload a **test CSV file** containing customer data "
    "with the target column **Exited**."
)

# ---------------------------------------
# Upload CSV
# ---------------------------------------

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# ---------------------------------------
# Model selection
# ---------------------------------------

model_name = st.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)

# ---------------------------------------
# Run prediction
# ---------------------------------------

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "Exited" not in df.columns:
        st.error("Uploaded file must contain 'Exited' column.")
        st.stop()

    y_true = df["Exited"]
    X = df.drop("Exited", axis=1)

    # Ensure correct feature order
    X = X[feature_names]
    X_scaled = scaler.transform(X)

    model = models[model_name]

    # y_prob = model.predict_proba(X_scaled)[:, 1]
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[:, 1]
    else:
        y_prob = model.predict(X_scaled)

    y_pred = (y_prob >= threshold).astype(int)

    # ---------------------------------------
    # Metrics
    # ---------------------------------------

    st.subheader("📊 Evaluation Metrics")

    st.write("Accuracy:", accuracy_score(y_true, y_pred))
    st.write("AUC:", roc_auc_score(y_true, y_prob))
    st.write("Precision:", precision_score(y_true, y_pred))
    st.write("Recall:", recall_score(y_true, y_pred))
    st.write("F1 Score:", f1_score(y_true, y_pred))
    st.write("MCC:", matthews_corrcoef(y_true, y_pred))

    # ---------------------------------------
    # Confusion Matrix
    # ---------------------------------------

    st.subheader("🔍 Confusion Matrix")
    st.write(confusion_matrix(y_true, y_pred))
