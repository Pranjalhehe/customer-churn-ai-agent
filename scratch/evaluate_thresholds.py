import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import load_and_clean_data
from src.features.build_features import build_features

def main():
    # Load dataset & model
    df = load_and_clean_data('data/raw/business_churn.csv')
    X, y = build_features(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    model = joblib.load('models/churn_model.pkl')
    y_scores = model.predict_proba(X_test)[:, 1]
    
    # Evaluate across thresholds 0.1 to 0.9
    thresholds = np.linspace(0.10, 0.90, 81)
    
    records = []
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        records.append({
            "threshold": round(t, 2),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f, 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        })
        
    res_df = pd.DataFrame(records)
    
    # 1. Threshold that maximizes F1-score
    best_f1_row = res_df.loc[res_df['f1_score'].idxmax()]
    
    # 2. Threshold that gives at least 70% recall while maximizing precision
    ge_70_recall = res_df[res_df['recall'] >= 0.70]
    if not ge_70_recall.empty:
        best_70_recall_row = ge_70_recall.loc[ge_70_recall['precision'].idxmax()]
    else:
        best_70_recall_row = res_df.loc[res_df['recall'].idxmax()]
        
    print("=== THRESHOLD ANALYSIS SUMMARY ===")
    print("\n--- Option A: Maximum F1-Score ---")
    print(best_f1_row.to_dict())
    
    print("\n--- Option B: Recall >= 70% with Max Precision ---")
    print(best_70_recall_row.to_dict())
    
    # Generate & Save Precision-Recall vs Threshold plot
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['threshold'], res_df['precision'], label='Precision', color='#10B981', linewidth=2.5)
    plt.plot(res_df['threshold'], res_df['recall'], label='Recall', color='#EF4444', linewidth=2.5)
    plt.plot(res_df['threshold'], res_df['f1_score'], label='F1 Score', color='#3B82F6', linewidth=2.5, linestyle='--')
    
    plt.axvline(x=best_f1_row['threshold'], color='#3B82F6', linestyle=':', label=f"Max F1 ({best_f1_row['threshold']:.2f})")
    plt.axvline(x=best_70_recall_row['threshold'], color='#EF4444', linestyle=':', label=f"Recall>=70% ({best_70_recall_row['threshold']:.2f})")
    
    plt.title('Precision, Recall, & F1-Score vs. Decision Threshold', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Decision Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xlim(0.10, 0.90)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='lower left')
    plt.tight_layout()
    
    chart_path = 'scratch/precision_recall_threshold_plot.png'
    plt.savefig(chart_path, dpi=300)
    print(f"\nSaved Precision-Recall threshold plot to {chart_path}")

if __name__ == '__main__':
    main()
