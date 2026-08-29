import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from src.data.preprocess import load_and_clean_data
from src.features.build_features import build_features

def verify():
    df = load_and_clean_data('data/raw/business_churn.csv')
    X, y = build_features(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    model = joblib.load('models/churn_model.pkl')
    y_pred = model.predict(X_test)
    
    print("=== VERIFYING SAVED MODEL (models/churn_model.pkl) ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"Precision (Churn): {precision_score(y_test, y_pred):.4f}")
    print(f"Recall (Churn):    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score (Churn):  {f1_score(y_test, y_pred):.4f}")
    print("\nFull Classification Report:\n", classification_report(y_test, y_pred, target_names=['No Churn (0)', 'Churn (1)']))
    
    with open('data/processed/dashboard_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print("\n=== FIRST 2 CUSTOMERS FROM data/processed/dashboard_data.json ===")
    print(json.dumps(data[:2], indent=2))

if __name__ == '__main__':
    verify()
