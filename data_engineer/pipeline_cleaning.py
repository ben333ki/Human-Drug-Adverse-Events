import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
import sys

# Import Config (ตรวจสอบว่าไฟล์ config.py อยู่โฟลเดอร์เดียวกันหรือ PYTHONPATH ถูกต้อง)
try:
    # ปรับ Import ให้เรียกใช้ DB_CONFIG ตามโครงสร้างใหม่
    from config import DB_CONFIG
except ImportError:
    print("⚠️ Error: config.py not found or missing variables. Please make sure config.py exists.")
    sys.exit(1)

# ==========================================
# 1. Configuration & Constants
# ==========================================

# Mapping ตามมาตรฐาน ICH E2B (R2) Code List
ROUTE_MAP = {
    # Common Routes
    '048': 'Oral',          '48': 'Oral',
    '058': 'Subcutaneous',  '58': 'Subcutaneous',   
    '042': 'Intravenous',   '42': 'Intravenous',    
    '065': 'Topical',       '65': 'Topical',        
    '061': 'Topical',       '61': 'Topical',        
    
    # Specific Codes
    '003': 'Cutaneous',     '3':  'Cutaneous',      
    '023': 'Intradermal',   '23': 'Intradermal',    
    '030': 'Intramuscular', '30': 'Intramuscular',  
    '047': 'Ophthalmic',    '47': 'Ophthalmic',     
    '051': 'Parenteral',    '51': 'Parenteral',     
    '059': 'Subdermal',     '59': 'Subdermal',      
    '060': 'Sublingual',    '60': 'Sublingual',     
    '062': 'Transdermal',   '62': 'Transdermal',    
    '064': 'Transplacental','64': 'Transplacental', 
    
    # Others
    '050': 'Rectal',        '50': 'Rectal',         
    '054': 'Rectal',        '54': 'Rectal',         
    '041': 'Inhalation',    '41': 'Inhalation',     
    '001': 'Auricular',     '1':  'Auricular',      
}

OUTPUT_DIR = "data_engineer/processed"
OUTPUT_CSV_FILENAME = "glp1_clean_final.csv"
TARGET_TABLE_NAME = "glp1_clean"

# ==========================================
# 2. Helper Functions
# ==========================================

def get_db_engine():
    """สร้าง Connection Engine ไปยัง PostgreSQL"""
    # ปรับมาใช้ค่าจาก Dictionary DB_CONFIG
    url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    return create_engine(url)

def convert_to_years(row):
    """แปลงอายุจากหน่วยต่างๆ ให้เป็น 'ปี' (Years)"""
    age = row['patient_age']
    unit = row['age_unit']
    
    if pd.isna(age) or pd.isna(unit):
        return np.nan
    
    try:
        age = float(age)
        unit = int(unit)
        
        if unit == 801:   return age            # Year
        elif unit == 802: return age / 12       # Month
        elif unit == 803: return age / 52       # Week
        elif unit == 804: return age / 365      # Day
        elif unit == 805: return age / 8760     # Hour
        elif unit == 800: return age * 10       # Decade
        else: return np.nan
    except:
        return np.nan

def clean_drug_route(val):
    """Standardize ข้อมูล Route of Administration"""
    if pd.isna(val) or str(val).strip() == "":
        return "Unknown"
    
    # Clean format: เอา 'Code_' และ '.0' ออก
    val_str = str(val).strip().replace("Code_", "")
    if val_str.endswith('.0'):
        val_str = val_str[:-2]
        
    # Map ข้อมูล
    if val_str in ROUTE_MAP:
        return ROUTE_MAP[val_str]
    
    # Fallback
    return val_str.title() if not val_str.isdigit() else f"Unknown_Code_{val_str}"

# ==========================================
# 3. Main Pipeline Logic
# ==========================================

def run_data_cleaning_pipeline():
    print("🚀 Starting Data Cleaning Pipeline...")
    
    try:
        engine = get_db_engine()
        source_table = DB_CONFIG['table_name']

        # --- Step 1: Load Data ---
        print(f"📥 Loading raw data from table: {source_table}")
        df = pd.read_sql(f"SELECT * FROM {source_table}", engine)
        print(f"   Initial Shape: {df.shape}")

        if df.empty:
            print("⚠️ Warning: Source table is empty. Aborting pipeline.")
            return

        # --- Step 2: Clean Seriousness ---
        print("🧹 Cleaning Seriousness...")
        df["seriousness"] = pd.to_numeric(df["seriousness"], errors='coerce').fillna(2)
        df["seriousness"] = df["seriousness"].map({1: 1, 2: 0}).astype(int)

        # --- Step 3: Clean Sex ---
        print("🧹 Cleaning Sex...")
        df["sex"] = df["sex"].astype("float").map({1.0: "M", 2.0: "F", 0.0: "Unknown"}).fillna("Unknown")

        # --- Step 4: Clean & Normalize Age ---
        print("🧹 Cleaning Age...")
        df['age_years'] = df.apply(convert_to_years, axis=1).round(2)
        # Filter valid age (0-120 years)
        before_filter = len(df)
        df = df[df['age_years'].between(0, 120)]
        print(f"   Dropped {before_filter - len(df)} rows with invalid age.")

        # --- Step 5: Clean Drug Route ---
        print("🧹 Cleaning Drug Route...")
        df["drug_route"] = df["drug_route"].apply(clean_drug_route)

        # --- Step 6: Clean Reaction & Dates ---
        print("🧹 Cleaning Reaction & Dates...")
        df["reaction"] = df["reaction"].astype(str).str.title().str.strip()
        df["receivedate"] = pd.to_datetime(df["receivedate"], format="%Y%m%d", errors="coerce")

        # --- Step 7: Final Validations ---
        print("🔍 Final Validation...")
        critical_cols = ["reaction", "seriousness", "age_years", "receivedate"]
        df_clean = df.dropna(subset=critical_cols).copy()
        
        # Reorder Columns
        final_cols_order = ['receivedate', 'drug_name', 'drug_route', 'sex', 'age_years', 'reaction', 'seriousness']
        available_cols = [c for c in final_cols_order if c in df_clean.columns]
        df_clean = df_clean[available_cols]

        print(f"   Original Shape: {df.shape}")
        print(f"   Final Clean Shape: {df_clean.shape}")

        # --- Step 8: Save Outputs ---
        
        # 8.1 Save to CSV
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILENAME)
        df_clean.to_csv(csv_path, index=False)
        print(f"✅ Saved CSV to: {csv_path}")

        # 8.2 Save to Database
        print(f"💾 Saving to PostgreSQL table: {TARGET_TABLE_NAME}...")
        
        df_clean.to_sql(TARGET_TABLE_NAME, engine, if_exists='replace', index=False)
        
        # Verify
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {TARGET_TABLE_NAME}"))
            count = result.scalar()
            print(f"✅ Saved successfully! Rows in DB: {count}")
            
        print("🎉 Pipeline Finished Successfully!")

    except Exception as e:
        print(f"❌ Pipeline Failed: {e}")
        sys.exit(1) # Raise error code for Airflow/Orchestrator

# ==========================================
# 4. Execution Entry Point
# ==========================================
if __name__ == "__main__":
    # บล็อกนี้จะทำงานเฉพาะเมื่อรันไฟล์นี้โดยตรง (เช่น python clean_data.py)
    # แต่ถ้า Airflow import run_data_cleaning_pipeline() บล็อกนี้จะไม่ทำงานซ้ำ
    run_data_cleaning_pipeline()