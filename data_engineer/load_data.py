# load_data.py
import pandas as pd
from sqlalchemy import create_engine
from config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, TABLE_NAME

RAW_CSV_PATH = "data_engineer/raw/openfda_raw.csv"

# Load CSV
df = pd.read_csv(RAW_CSV_PATH)

# Connect PostgreSQL
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Save to table
df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
print(f"Saved {len(df)} records to PostgreSQL table '{TABLE_NAME}'")
