import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ตั้งค่าหน้าเว็บ (Page Config)
st.set_page_config(
    page_title="GLP-1 Safety Predictor",
    page_icon="💊",
    layout="centered"
)

# ----------------------------------------------
# 1. Load Model & Columns
# ----------------------------------------------
@st.cache_resource
def load_artifacts():
    # ตรวจสอบว่ามีไฟล์โมเดลจริงไหม
    if not os.path.exists('data_science/models/glp1_risk_predictor.pkl'):
        st.error("❌ Model file not found. Please run the training script first.")
        return None, None
    
    model = joblib.load('data_science/models/glp1_risk_predictor.pkl')
    model_columns = joblib.load('data_science/models/model_columns.pkl')
    return model, model_columns

model, model_columns = load_artifacts()

# ----------------------------------------------
# 2. UI Design (User Interface)
# ----------------------------------------------
st.title("💊 GLP-1 Safety Predictor")
st.markdown("""
เครื่องมือประเมินความเสี่ยงในการเกิด **อาการข้างเคียงรุนแรง (Serious Adverse Event)** จากการใช้ยากลุ่ม GLP-1 Agonists โดยใช้ AI Machine Learning
""")

st.divider() # เส้นขีดคั่น

col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Info")
    age = st.number_input("Age (Years)", min_value=0, max_value=120, value=50)
    sex = st.radio("Sex", ["Female", "Male", "Unknown"], horizontal=True)

with col2:
    st.subheader("Drug Info")
    # รายชื่อยาต้องตรงกับที่ใช้ Train (ตัวพิมพ์ใหญ่ตามข้อมูลจริง)
    drug_list = ['ZEPBOUND', 'SEMAGLUTIDE', 'LIRAGLUTIDE', 'DULAGLUTIDE', 'TIRZEPATIDE']
    drug_name = st.selectbox("Select Drug", drug_list)

# ----------------------------------------------
# 3. Prediction Logic
# ----------------------------------------------
if st.button("Analyze Risk", type="primary", use_container_width=True):
    if model is not None:
        # A. เตรียมกระดานข้อมูลเปล่าๆ (DataFrame) ที่มีคอลัมน์ครบตามตอนเทรน
        input_data = pd.DataFrame(columns=model_columns)
        input_data.loc[0] = 0  # เติม 0 ให้หมดก่อน (Default)

        # B. กรอกข้อมูล Age (ใส่ง่ายๆ เลยเพราะเป็นตัวเลขอยู่แล้ว)
        if 'age_years' in input_data.columns:
            input_data['age_years'] = age

        # C. กรอกข้อมูล Sex (One-Hot Mapping)
        # ตอนเทรนเรามี sex_M, sex_Unknown (sex_F ถูกตัดออกเป็น base case)
        if sex == 'Male' and 'sex_M' in input_data.columns:
            input_data['sex_M'] = 1
        elif sex == 'Unknown' and 'sex_Unknown' in input_data.columns:
            input_data['sex_Unknown'] = 1
        # ถ้าเป็น Female ไม่ต้องทำอะไร ปล่อยให้ sex_M=0, sex_Unknown=0 ถูกแล้ว

        # D. กรอกข้อมูล Drug (One-Hot Mapping)
        # สร้างชื่อคอลัมน์เป้าหมาย เช่น 'drug_name_ZEPBOUND'
        target_drug_col = f"drug_name_{drug_name}"
        if target_drug_col in input_data.columns:
            input_data[target_drug_col] = 1

        # E. ทำนายผล (Predict)
        try:
            # ความน่าจะเป็น (Probability) ว่าจะเป็น Serious (Class 1)
            prob = model.predict_proba(input_data)[0][1]
            
            st.divider()
            st.subheader("Analysis Result")
            
            # Gauge Chart แบบง่าย (Progress Bar)
            st.write("Risk Probability (โอกาสเกิดเคสรุนแรง)")
            st.progress(prob)
            st.caption(f"Confidence Score: {prob*100:.2f}%")

            # สรุปผล
            if prob > 0.5:
                st.error(f"⚠️ **High Risk** (มีความเสี่ยงสูง)")
                st.write(f"จากการวิเคราะห์ ผู้ป่วยรายนี้มีความเสี่ยงสูงที่จะเกิดอาการข้างเคียงรุนแรงเมื่อใช้ยา **{drug_name}**")
            else:
                st.success(f"✅ **Low Risk** (ความเสี่ยงต่ำ)")
                st.write(f"จากการวิเคราะห์ คาดการณ์ว่าเป็นอาการข้างเคียงทั่วไป (Non-Serious) สำหรับยา **{drug_name}**")
                
            # (Optional) แสดง Data ที่ส่งเข้าโมเดลเพื่อ Debug
            with st.expander("See Technical Details"):
                st.write("Input Data Vector sent to Model:")
                st.dataframe(input_data)
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")

else:
    st.info("กรุณากรอกข้อมูลและกดปุ่ม Analyze Risk เพื่อเริ่มการวิเคราะห์")

# Footer
st.markdown("---")
st.caption("Model: Random Forest Classifier (ROC-AUC 0.93)")