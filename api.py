from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os

# 1. Load Model & Columns (โหลดครั้งเดียวตอนเริ่ม Server)
# ตรวจสอบว่าไฟล์โมเดลมีอยู่จริงไหม
if not os.path.exists('data_science/models/glp1_risk_predictor.pkl'):
    raise RuntimeError("❌ Model file not found. Please train model first.")

model = joblib.load('data_science/models/glp1_risk_predictor.pkl')
model_columns = joblib.load('data_science/models/model_columns.pkl')

# 2. Setup FastAPI App
app = FastAPI(
    title="GLP-1 Risk Prediction API",
    description="API สำหรับประเมินความเสี่ยงอาการข้างเคียงยา GLP-1",
    version="1.0.0"
)

# 3. Define Input Structure (ตรวจสอบ Data Type อัตโนมัติ)
class PatientData(BaseModel):
    age_years: int
    sex: str  # คาดหวัง: "Male", "Female", "Unknown"
    drug_name: str # คาดหวังชื่อยาภาษาอังกฤษตัวพิมพ์ใหญ่

# 4. Create Prediction Endpoint
@app.post("/predict", tags=["Prediction"])
def predict_risk(data: PatientData):
    try:
        # เตรียม DataFrame ว่างๆ ให้เหมือนตอนเทรน
        input_df = pd.DataFrame(columns=model_columns)
        input_df.loc[0] = 0
        
        # --- Logic การเติมข้อมูล (Mapping) ---
        
        # 1. Age
        if 'age_years' in input_df.columns:
            input_df['age_years'] = data.age_years
            
        # 2. Sex
        if data.sex == 'Male' and 'sex_M' in input_df.columns:
            input_df['sex_M'] = 1
        elif data.sex == 'Unknown' and 'sex_Unknown' in input_df.columns:
            input_df['sex_Unknown'] = 1
            
        # 3. Drug Name
        target_drug = f"drug_name_{data.drug_name.upper()}"
        if target_drug in input_df.columns:
            input_df[target_drug] = 1
        else:
            # ถ้าชื่อยาที่ส่งมา ไม่ตรงกับที่มีในโมเดล
            return {
                "warning": f"Drug '{data.drug_name}' not found in training set. Prediction might be inaccurate.",
                "risk_probability": None
            }
            
        # --- Prediction ---
        prob = model.predict_proba(input_df)[0][1]
        
        return {
            "status": "success",
            "input_summary": {
                "age": data.age_years,
                "drug": data.drug_name
            },
            "prediction": {
                "risk_probability": float(prob), # แปลง numpy float เป็น python float
                "risk_level": "High" if prob > 0.5 else "Low",
                "is_serious_event_likely": bool(prob > 0.5)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health Check Endpoint (สำหรับเช็คว่า Server ตายไหม)
@app.get("/")
def health_check():
    return {"status": "running", "model_loaded": True}

