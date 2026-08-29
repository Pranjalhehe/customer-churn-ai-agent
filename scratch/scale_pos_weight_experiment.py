import os
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import load_and_clean_data
from src.features.build_features import build_features

def run_experiment():
    df = load_and_clean_data('data/raw/business_churn.csv')
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    n0 = float((y_train == 0).sum())
    n1 = float((y_train == 1).sum())
    full_ratio = n0 / n1
    half_ratio = 0.5 * full_ratio
    sqrt_ratio = np.sqrt(full_ratio)
    
    threshold = 0.44
    
    experiments = [
        ("Full Ratio (majority / minority)", full_ratio),
        ("Half Ratio (0.5 * full_ratio)", half_ratio),
        ("Sqrt Ratio (sqrt(full_ratio))", sqrt_ratio)
    ]
    
    results = []
    
    for name, weight in experiments:
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            scale_pos_weight=weight,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= threshold).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f = f1_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            "variant": name,
            "weight": round(weight, 4),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f, 4),
            "accuracy": round(acc, 4)
        })
        
    print("=== SCALE_POS_WEIGHT EXPERIMENT RESULTS (Threshold = 0.44) ===")
    for r in results:
        print(r)
        
    res_df = pd.DataFrame(results)
    res_df.to_json('scratch/scale_pos_weight_results.json', orient='records', indent=2)

if __name__ == '__main__':
    run_experiment()
