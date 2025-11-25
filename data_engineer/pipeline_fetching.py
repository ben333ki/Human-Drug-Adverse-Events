import requests
import pandas as pd
import time
import os
import sys
from sqlalchemy import create_engine, text

# ตรวจสอบการ Import Config
try:
    from config import (
        OPENFDA_URL, LIMIT, GLP1_DRUGS, RAW_DATA_PATH, API_KEY,
        DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, TABLE_NAME
    )
except ImportError:
    print("⚠️ Error: config.py not found or missing variables.")
    sys.exit(1)

# ==========================================
# 1. Extract Function (Fetch from API)
# ==========================================

def fetch_openfda_multi_drugs():
    """
    ดึงข้อมูลจาก OpenFDA API ตามรายชื่อยาที่กำหนด
    Returns: DataFrame ที่มีข้อมูลดิบ
    """
    print(f"🚀 Starting Extraction Process...")
    all_records = []

    for drug in GLP1_DRUGS:
        print(f"   Fetching data for: {drug} ...")
        skip = 0
        
        while True:
            # Query Construction
            search_query = f'patient.drug.medicinalproduct:"{drug}"'
            url = f"{OPENFDA_URL}?api_key={API_KEY}&limit={LIMIT}&skip={skip}&search={search_query}"
            
            try:
                resp = requests.get(url, timeout=30) # เพิ่ม timeout กันค้าง
            except requests.exceptions.RequestException as e:
                print(f"❌ Network Error for {drug}: {e}")
                break

            if resp.status_code != 200:
                print(f"   ⚠️ Finished or Stop at {drug} (Status: {resp.status_code})")
                break

            data = resp.json()
            results = data.get("results", [])
            
            if not results:
                break

            # Flattening Logic
            for r in results:
                patient = r.get("patient", {})
                drugs = patient.get("drug", [])
                reactions = patient.get("reaction", [])
                
                for d in drugs:
                    drug_name = d.get("medicinalproduct")
                    
                    # กรองเฉพาะยาที่เราสนใจ (เพราะ API อาจคืนยาอื่นที่ผู้ป่วยกินร่วมด้วย)
                    if drug_name not in GLP1_DRUGS:
                        continue
                        
                    for rxn in reactions:
                        all_records.append({
                            "patient_age": patient.get("patientonsetage"),
                            "age_unit": patient.get("patientonsetageunit"),
                            "sex": patient.get("patientsex"),
                            "drug_name": drug_name,
                            "drug_route": d.get("drugadministrationroute"),
                            "reaction": rxn.get("reactionmeddrapt"),
                            "seriousness": r.get("serious"),
                            "receivedate": r.get("receivedate")
                        })

            skip += LIMIT
            time.sleep(0.1)  # Rate Limit protection

    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Save Raw Data to CSV (Data Lake Layer)
    if not df.empty:
        # สร้าง Folder โดยอัตโนมัติถ้ายังไม่มี
        output_dir = os.path.dirname(RAW_DATA_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"✅ Extracted {len(df)} records. Saved raw CSV at: {RAW_DATA_PATH}")
    else:
        print("⚠️ No data fetched.")

    return df

# ==========================================
# 2. Load Function (Save to PostgreSQL)
# ==========================================

def load_data_to_postgres(df):
    """
    รับ DataFrame และบันทึกลง PostgreSQL (Raw Table)
    """
    if df.empty:
        print("⚠️ DataFrame is empty. Skipping Load step.")
        return

    print(f"💾 Loading data to PostgreSQL table: '{TABLE_NAME}' ...")
    
    try:
        # Create Connection String
        url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(url)

        # Write to DB
        # if_exists='replace': สำหรับ Raw Data เรามักจะ Truncate แล้วลงใหม่ หรือ 'append' แล้วแต่นโยบาย
        df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
        
        # Verify Count
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}"))
            count = result.scalar()
            print(f"✅ Successfully loaded {count} records to DB.")
            
    except Exception as e:
        print(f"❌ Error loading data to DB: {e}")
        raise e

# ==========================================
# 3. Main Pipeline Runner
# ==========================================

def run_fetching_pipeline():
    """
    Orchestrate the Fetch -> Load process
    """
    print("="*40)
    print("      STARTING INGESTION PIPELINE      ")
    print("="*40)
    
    # Step 1: Extract
    df_raw = fetch_openfda_multi_drugs()
    
    # Step 2: Load
    if not df_raw.empty:
        load_data_to_postgres(df_raw)
    
    print("="*40)
    print("      PIPELINE COMPLETED      ")
    print("="*40)

if __name__ == "__main__":
    run_fetching_pipeline()