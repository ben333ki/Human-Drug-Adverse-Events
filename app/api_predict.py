# app/api_predict.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# -----------------------------
# Load trained model & features
# -----------------------------
with open("models/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models/rf_features.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Adverse Event Prediction API")

# -----------------------------
# Input schema
# -----------------------------
class InputData(BaseModel):
    # ตัวอย่าง columns (ต้องตรงกับ rf_features)
    age_years: float
    sex_FEMALE: int = 0
    sex_UNKNOWN: int = 0
    drug_route_ORAL: int = 0
    drug_route_UNKNOWN: int = 0
    reaction_freq: float
    drug_name_freq: float
    # ... เพิ่ม columns ที่เหลือให้ครบตาม feature_cols

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    # ตรวจ columns ครบ
    missing_cols = [c for c in feature_cols if c not in df.columns]
    for c in missing_cols:
        df[c] = 0  # เติม 0 ให้คอลัมน์ที่ขาด

    df = df[feature_cols]  # เรียงคอลัมน์ตามโมเดล
    pred = rf_model.predict(df)
    prob = rf_model.predict_proba(df)
    return {
        "prediction": int(pred[0]),
        "probability": prob[0].tolist()
    }
