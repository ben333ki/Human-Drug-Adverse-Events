# OpenFDA API
OPENFDA_URL = "https://api.fda.gov/drug/event.json"
API_KEY = "wgUCZbR2UAltcSJ8z0oLswpB7W6ZiQm6vctwaEv6"
LIMIT = 100  # ดึงทีละ 100 records

# ดึงหลาย GLP-1 drugs
GLP1_DRUGS = ["ZEPBOUND", "SEMAGLUTIDE", "LIRAGLUTIDE", "DULAGLUTIDE", "TIRZEPATIDE"]

RAW_DATA_PATH = "data_engineer/raw/openfda_raw.csv"

# PostgreSQL connection
DB_USER = "postgres"
DB_PASSWORD = "2306"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "glp1_adverse_db"
TABLE_NAME = "adverse_events"
