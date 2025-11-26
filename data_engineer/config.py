import os
from dotenv import load_dotenv

# โหลดค่าจากไฟล์ .env
load_dotenv()

# --- OpenFDA Configuration ---
OPENFDA_URL = "https://api.fda.gov/drug/event.json"
# ดึง API Key จาก .env (ถ้าไม่มีให้เป็น None หรือใส่ Default)
API_KEY = os.getenv("OPENFDA_API_KEY") 
LIMIT = 100 

# List เก็บไว้ใน Python จะจัดการง่ายกว่าใน .env
GLP1_DRUGS = [
    "ZEPBOUND", 
    "SEMAGLUTIDE", 
    "LIRAGLUTIDE", 
    "DULAGLUTIDE", 
    "TIRZEPATIDE"
]

RAW_DATA_PATH = os.path.join("data_engineer", "raw", "openfda_raw.csv")

# --- Database Connection Configuration ---
DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"), # ถ้าไม่เจอให้ใช้ localhost เป็นค่า default
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "glp1_adverse_db"),
    "table_name": "adverse_events"
}
