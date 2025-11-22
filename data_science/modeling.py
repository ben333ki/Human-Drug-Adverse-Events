import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import os

def main():
    FEATURE_PATH = "data_engineer/processed/glp1_features_fixed.csv"
    df = pd.read_csv(FEATURE_PATH)

    y = df['seriousness']
    X = df.drop(columns=['seriousness'])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )

    # RandomForest
    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("RandomForest Performance:")
    print(classification_report(y_test, y_pred_rf))

    # บันทึกโมเดลและ feature columns
    os.makedirs("models", exist_ok=True)
    with open("models/rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open("models/rf_features.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)
    print("RandomForest model and feature columns saved!")

    # XGBoost (สามารถบันทึกได้เช่นกันถ้าต้องการ)
    pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42,
        scale_pos_weight=pos_weight, eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    print("XGBoost Performance:")
    print(classification_report(y_test, y_pred_xgb))

    # Logistic Regression
    lr = LogisticRegression(
        max_iter=1000, class_weight='balanced', solver='liblinear'
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("Logistic Regression Performance:")
    print(classification_report(y_test, y_pred_lr))

    # Confusion Matrices
    for model, y_pred in zip(
        ["RandomForest","XGBoost","LogisticRegression"],
        [y_pred_rf, y_pred_xgb, y_pred_lr]
    ):
        print(f"Confusion Matrix — {model}")
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.show()

if __name__ == "__main__":
    main()
