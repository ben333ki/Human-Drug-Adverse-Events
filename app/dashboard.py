# dashboard.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import classification_report

# -----------------------------
# Load feature dataset
# -----------------------------
FEATURE_PATH = "data_engineer/processed/glp1_features_fixed.csv"
df = pd.read_csv(FEATURE_PATH)

# -----------------------------
# Load model
# -----------------------------
with open("models/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models/rf_features.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# -----------------------------
# Sidebar for prediction
# -----------------------------
st.sidebar.header("Predict Adverse Event Severity")
input_data = {}
for col in feature_cols:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_cols]
    pred = rf_model.predict(input_df)
    prob = rf_model.predict_proba(input_df)
    st.sidebar.success(f"Prediction: {int(pred[0])} (0=non-serious, 1=serious)")
    st.sidebar.info(f"Probability: {prob[0]}")

# -----------------------------
# Main dashboard
# -----------------------------
st.title("Human Drug Adverse Event Dashboard")

# Target distribution
st.subheader("Seriousness Distribution")
st.bar_chart(df['seriousness'].value_counts(normalize=True))

# Feature correlations
st.subheader("Feature Correlations with Seriousness")
corr = df.corr()['seriousness'].sort_values(ascending=False)
st.dataframe(corr.head(10))

# Top features from RandomForest
st.subheader("Top Feature Importances (RandomForest)")
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
st.bar_chart(importances.sort_values(ascending=False).head(20))
