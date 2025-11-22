# feature_engineering_fixed.py
import pandas as pd
import os

INPUT_CSV = "data_engineer/processed/glp1_clean.csv"
OUTPUT_CSV = "data_engineer/processed/glp1_features_fixed.csv"

# -----------------------------
# 1) Load cleaned data
# -----------------------------
df = pd.read_csv(INPUT_CSV)
print("Cleaned data shape:", df.shape)

# -----------------------------
# 2) Convert age to years
# -----------------------------
def convert_age(row):
    if row['age_unit'] in [800, 801, '800', '801']:  # years
        return row['patient_age']
    elif row['age_unit'] in [802, '802']:  # months
        return row['patient_age'] / 12
    elif row['age_unit'] in [803, '803']:  # weeks
        return row['patient_age'] / 52
    elif row['age_unit'] in [804, '804']:  # days
        return row['patient_age'] / 365
    else:
        return row['patient_age']

df['age_years'] = df.apply(convert_age, axis=1)

# -----------------------------
# 3) Fill missing categorical
# -----------------------------
df['sex'] = df['sex'].fillna('UNKNOWN')
df['drug_route'] = df['drug_route'].fillna('UNKNOWN')
df['drug_name'] = df['drug_name'].fillna('UNKNOWN')

# -----------------------------
# 4) Frequency encoding for high-cardinality columns
# -----------------------------
for col in ['reaction', 'drug_name']:
    freq = df[col].value_counts(normalize=True)
    df[f'{col}_freq'] = df[col].map(freq)

# -----------------------------
# 5) One-hot encode low-cardinality columns
# -----------------------------
df = pd.get_dummies(df, columns=['sex','drug_route'], drop_first=True)

# -----------------------------
# 6) Prepare features and target
# -----------------------------
y = df['seriousness']
X = df.drop(columns=['seriousness','patient_age','age_unit','reaction','drug_name','receivedate'])

print("Features shape:", X.shape)
print("Target distribution:\n", y.value_counts(normalize=True))

# -----------------------------
# 7) Save feature dataset
# -----------------------------
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
X['seriousness'] = y
X.to_csv(OUTPUT_CSV, index=False)
print(f"Saved feature dataset -> {OUTPUT_CSV}")
