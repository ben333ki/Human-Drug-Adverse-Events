# explainability_insights.py
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

# -----------------------------
# 1) Load feature dataset
# -----------------------------
FEATURE_PATH = "data_engineer/processed/glp1_features.csv"
df = pd.read_csv(FEATURE_PATH)

print("Full dataset shape (numeric):", df.shape)

# -----------------------------
# 2) Separate features and target
# -----------------------------
target_col = 'label' if 'label' in df.columns else 'seriousness'
y = df[target_col]
X = df.drop(columns=[target_col])

# -----------------------------
# 3) Sample dataset for fast SHAP
# -----------------------------
SAMPLE_SIZE = 10000
if len(df) > SAMPLE_SIZE:
    np.random.seed(42)
    sample_idx = np.random.choice(df.index, size=SAMPLE_SIZE, replace=False)
    X_sample = X.loc[sample_idx].reset_index(drop=True)
    y_sample = y.loc[sample_idx].reset_index(drop=True)
    print(f"Using sample dataset for explainability: {X_sample.shape}")
else:
    X_sample = X.copy()
    y_sample = y.copy()
    print("Using full dataset for explainability.")

# -----------------------------
# 4) Train RandomForest on sample
# -----------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf.fit(X_sample, y_sample)
print("RandomForest trained on sample dataset.")

# -----------------------------
# 5) SHAP values
# -----------------------------
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)

# Binary classification check
if isinstance(shap_values, list) and len(shap_values) == 2:
    sv = shap_values[1]  # class 1 (serious)
else:
    sv = shap_values

print("SHAP values shape:", sv.shape)

# -----------------------------
# 6) Summary plot
# -----------------------------
shap.summary_plot(sv, X_sample, show=True)

# -----------------------------
# 7) Feature importance (mean absolute SHAP)
# -----------------------------
shap_mean_abs = np.abs(sv).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': X_sample.columns,
    'mean_abs_shap': shap_mean_abs
}).sort_values(by='mean_abs_shap', ascending=False)
print("\nTop 10 important features (SHAP):")
print(feature_importance.head(10))

# -----------------------------
# 8) Partial Dependence Plots (PDP)
# -----------------------------
top_features = feature_importance['feature'].head(3).tolist()
print("\nGenerating PDP for top 3 features:", top_features)

for feat in top_features:
    PartialDependenceDisplay.from_estimator(rf, X_sample, [feat], kind='average')
    plt.show()

print("\nPhase 5 complete: SHAP, top features, and PDP generated.")
