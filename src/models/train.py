import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from src.data.preprocess import load_and_clean_data
from src.features.build_features import build_features

def train_churn_model(data_path: str = 'data/raw/business_churn.csv', model_output_path: str = 'models/churn_model.pkl'):
    """
    Train XGBoost model on business churn dataset using scale_pos_weight for native class imbalance handling.
    """
    print(f"Loading and preprocessing data from {data_path}...")
    df_clean = load_and_clean_data(data_path)
    X, y = build_features(df_clean)
    
    # 1. Stratified train/test split (80/20) with fixed seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Original Train Churn Distribution:\n{y_train.value_counts()}")
    
    # 2. Calculate scale_pos_weight = (count of class 0) / (count of class 1)
    n_class0 = (y_train == 0).sum()
    n_class1 = (y_train == 1).sum()
    scale_weight = float(n_class0) / float(n_class1)
    print(f"Calculated scale_pos_weight: {scale_weight:.4f}")
    
    # 3. Train XGBoost classifier with scale_pos_weight
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # 4. Evaluate on untouched test set
    y_pred = model.predict(X_test)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n================ EVALUATION ON UNTOUCHED TEST SET ================")
    print(f"Confusion Matrix:\n{cm}")
    print("\n--- Key Churn Metrics (Class 1) ---")
    print(f"Precision (Churn): {precision:.4f}")
    print(f"Recall (Churn):    {recall:.4f}")
    print(f"F1-Score (Churn):  {f1:.4f}")
    
    print("\n--- Full Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['No Churn (0)', 'Churn (1)']))
    
    # 5. Save trained model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Successfully saved trained XGBoost model to {model_output_path}")

if __name__ == '__main__':
    train_churn_model()
