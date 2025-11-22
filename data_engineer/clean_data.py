# clean_data.py
import pandas as pd
from sqlalchemy import create_engine
from config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, TABLE_NAME
import os

OUTPUT_PATH = "data_engineer/processed/glp1_clean.csv"

# 1) Load from PostgreSQL
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)

print("Raw shape:", df.shape)


# ---------------------------
# 2) CLEANING
# ---------------------------

# --- seriousness ---
df["seriousness"] = df["seriousness"].astype(int)
df["seriousness"] = df["seriousness"].map({1: 1, 2: 0})


# --- sex ---
df["sex"] = df["sex"].astype("float").map({
    1.0: "M",
    2.0: "F",
    0.0: None
})


# --- patient_age ---
df["patient_age"] = pd.to_numeric(df["patient_age"], errors="coerce")
df = df[df["patient_age"].between(0, 120)]


# --- age_unit ---
age_unit_map = {
    801.0: "years",
    802.0: "months",
    803.0: "weeks",
    804.0: "days",
}

df["age_unit"] = df["age_unit"].map(age_unit_map)
df["age_unit"] = df["age_unit"].fillna("unknown")


# --- drug_route ---
df["drug_route"] = df["drug_route"].fillna("").astype(str).str.upper()
df.loc[df["drug_route"].str.strip() == "", "drug_route"] = "UNKNOWN"


# --- reaction ---
df["reaction"] = df["reaction"].astype(str).str.title()


# --- receivedate ---
df["receivedate"] = pd.to_datetime(df["receivedate"], format="%Y%m%d", errors="coerce")


# --- drop rows missing important info ---
df = df.dropna(subset=["reaction", "seriousness", "patient_age"])

print("Cleaned shape:", df.shape)


# ---------------------------
# 3) SAVE CLEAN DATASET
# ---------------------------
os.makedirs("data_engineer/processed", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved cleaned CSV -> {OUTPUT_PATH}")


# ---------------------------
# 4) Save back to PostgreSQL
# ---------------------------
df.to_sql("glp1_clean", engine, if_exists="replace", index=False)
print("Saved cleaned data -> PostgreSQL table 'glp1_clean'")
