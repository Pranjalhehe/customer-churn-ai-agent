import os
import sys
import joblib
import shap
import pandas as pd
import numpy as np

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import load_and_clean_data
from src.features.build_features import build_features

def analyze_nps():
    m = joblib.load('models/churn_model.pkl')
    raw_df = pd.read_csv('demo_samples/fresh_customers.csv')
    clean_df = load_and_clean_data('demo_samples/fresh_customers.csv')
    X, y = build_features(clean_df)
    X = X.reindex(columns=m.feature_names_in_, fill_value=0)
    
    explainer = shap.TreeExplainer(m)
    
    row7 = X.iloc[[6]].copy()
    raw_row7 = raw_df.iloc[6]
    
    print("=== CUSTOMER DEMO_00007 RAW METRICS ===")
    print(raw_row7[['customer_id', 'nps_score', 'csat_score', 'monthly_logins', 'tenure_months', 'last_login_days_ago', 'churn']])
    
    print("\n=== SHAP VALUES FOR DEMO_00007 ACROSS DIFFERENT NPS SCORES ===")
    nps_col_idx = X.columns.get_loc('nps_score')
    
    for test_nps in [-100, -50, -33, -10, 0, 10, 50, 100]:
        row_test = row7.copy()
        row_test['nps_score'] = test_nps
        prob = m.predict_proba(row_test)[0, 1]
        shap_vector = explainer(row_test).values
        if len(shap_vector.shape) == 3:
            s_val = shap_vector[0, nps_col_idx, 1]
        else:
            s_val = shap_vector[0, nps_col_idx]
        print(f"NPS = {test_nps:>4}: Churn Prob = {prob:.4f}, NPS SHAP = {s_val:+.4f}")

    print("\n=== SHAP VALUES FOR NPS SCORE ACROSS ALL 20 DEMO CUSTOMERS ===")
    all_shap = explainer(X).values
    if len(all_shap.shape) == 3:
        all_nps_shap = all_shap[:, nps_col_idx, 1]
    else:
        all_nps_shap = all_shap[:, nps_col_idx]
        
    for i in range(len(X)):
        cid = raw_df.iloc[i]['customer_id']
        nps = raw_df.iloc[i]['nps_score']
        prob = m.predict_proba(X.iloc[[i]])[0, 1]
        s_val = all_nps_shap[i]
        print(f"{cid}: NPS = {nps:>4}, Churn Prob = {prob:.4f}, NPS SHAP = {s_val:+.4f}")

if __name__ == '__main__':
    analyze_nps()
