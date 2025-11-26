import streamlit as st
import requests
import pandas as pd

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="GLP-1 Safety Predictor",
    page_icon="💊",
    layout="centered"
)

# ----------------------------------------------
# 1. UI Design
# ----------------------------------------------
st.title("💊 GLP-1 Safety Predictor")
st.markdown("""
เครื่องมือประเมินความเสี่ยงในการเกิด **อาการข้างเคียงรุนแรง (Serious Adverse Event)** จากการใช้ยากลุ่ม GLP-1 Agonists โดยใช้ AI Machine Learning (เชื่อมต่อผ่าน API)
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Info")
    age = st.number_input("Age (Years)", min_value=0, max_value=120, value=50)
    sex = st.radio("Sex", ["Female", "Male", "Unknown"], horizontal=True)

with col2:
    st.subheader("Drug Info")
    drug_list = ['ZEPBOUND', 'SEMAGLUTIDE', 'LIRAGLUTIDE', 'DULAGLUTIDE', 'TIRZEPATIDE']
    drug_name = st.selectbox("Select Drug", drug_list)

# ----------------------------------------------
# 2. Prediction Logic (Call API)
# ----------------------------------------------
# URL ของ API ที่คุณ Deploy บน Render
API_URL = "https://fda-risk-api.onrender.com/predict"

if st.button("Analyze Risk", type="primary", use_container_width=True):
    
    # เตรียมข้อมูลส่งไป API (JSON Payload)
    payload = {
        "age_years": int(age),
        "sex": sex,
        "drug_name": drug_name
    }
    
    with st.spinner('🤖 AI is analyzing... please wait'):
        try:
            # ยิง Request ไปที่ Render
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # ดึงค่าจากผลลัพธ์
                pred = result.get("prediction", {})
                prob = pred.get("risk_probability", 0)
                risk_level = pred.get("risk_level", "Unknown")
                
                st.divider()
                st.subheader("Analysis Result")
                
                # แสดงผล
                st.write("Risk Probability (โอกาสเกิดเคสรุนแรง)")
                st.progress(prob)
                st.caption(f"Confidence Score: {prob*100:.2f}%")
                
                if prob > 0.5:
                    st.error(f"⚠️ **{risk_level} Risk** (มีความเสี่ยงสูง)")
                    st.write(f"จากการวิเคราะห์ ผู้ป่วยรายนี้มีความเสี่ยงสูงที่จะเกิดอาการข้างเคียงรุนแรงเมื่อใช้ยา **{drug_name}**")
                else:
                    st.success(f"✅ **{risk_level} Risk** (ความเสี่ยงต่ำ)")
                    st.write(f"จากการวิเคราะห์ คาดการณ์ว่าเป็นอาการข้างเคียงทั่วไป (Non-Serious) สำหรับยา **{drug_name}**")
                    
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            st.error(f"Connection Failed: {e}")
            st.info("💡 สาเหตุอาจเกิดจาก Server บน Render กำลังตื่น (Cold Start) กรุณารอสักครู่แล้วลองกดใหม่")

else:
    st.info("👈 กรุณากรอกข้อมูลและกดปุ่ม Analyze Risk เพื่อเริ่มการวิเคราะห์")

# Footer
st.markdown("---")
st.caption(f"Backend API: {API_URL}")