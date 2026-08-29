import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import load_and_clean_data
from src.features.build_features import build_features
from src.models.explain import ChurnExplainer, get_top_risk_factors
from src.models.predict import DECISION_THRESHOLD
from src.agent.risk_explainer import explain_risk
from src.agent.retention_actions import recommend_actions

def get_risk_level(prob: float, threshold: float = DECISION_THRESHOLD) -> str:
    """
    Categorize churn probability into High, Medium, or Low risk level based on production 0.44 threshold:
    - High: prob >= 0.44 (Classified as Churn)
    - Medium: 0.20 <= prob < 0.44
    - Low: prob < 0.20
    """
    if prob >= threshold:
        return "High"
    elif prob >= 0.20:
        return "Medium"
    else:
        return "Low"

def get_customer_risk_profile(customer_row: pd.DataFrame, customer_id: Optional[str] = None, explainer: Optional[ChurnExplainer] = None) -> Dict[str, Any]:
    """
    End-to-end pipeline for a single customer:
    1. Compute churn probability and SHAP risk factors
    2. Label risk_level using production 0.44 threshold
    3. Generate natural language explanation using LLM agent
    4. Generate actionable retention recommendations
    """
    if isinstance(customer_row, pd.Series):
        customer_row = customer_row.to_frame().T
        
    if explainer is not None:
        shap_output = explainer.get_top_risk_factors(customer_row, top_n=3)
    else:
        shap_output = get_top_risk_factors(customer_row, top_n=3)
        
    prob = shap_output.get("churn_probability", 0.0)
    risk_lvl = get_risk_level(prob)
    
    top_factors = shap_output.get("top_risk_factors", [])
    explanation = explain_risk(shap_output)
    actions = recommend_actions(explanation, top_risk_factors=top_factors, risk_level=risk_lvl)
    
    return {
        "customer_id": customer_id if customer_id is not None else "N/A",
        "churn_probability": prob,
        "churn_prediction": 1 if prob >= DECISION_THRESHOLD else 0,
        "risk_level": risk_lvl,
        "threshold_used": DECISION_THRESHOLD,
        "top_risk_factors": top_factors,
        "explanation": explanation,
        "recommended_actions": actions
    }

def get_at_risk_customers(X_data: pd.DataFrame, customer_ids: Optional[pd.Series] = None, top_n: int = 10, model_path: str = 'models/churn_model.pkl') -> List[Dict[str, Any]]:
    """
    Evaluates dataset X_data, finds top N customers highest at risk of churn,
    and returns full risk profiles for each.
    """
    explainer = ChurnExplainer(model_path)
    probs = explainer.model.predict_proba(X_data)[:, 1]
    
    # Sort indices by probability descending
    sorted_indices = np.argsort(-probs)[:top_n]
    
    results = []
    for idx in sorted_indices:
        row = X_data.iloc[[idx]]
        c_id = str(customer_ids.iloc[idx]) if customer_ids is not None else f"Cust_{idx}"
        profile = get_customer_risk_profile(row, customer_id=c_id, explainer=explainer)
        results.append(profile)
        
    return results

def get_sampled_risk_customers(
    X_data: pd.DataFrame, 
    customer_ids: Optional[pd.Series] = None, 
    n_high: int = 10,
    n_medium: int = 5,
    n_low: int = 3,
    model_path: str = 'models/churn_model.pkl'
) -> List[Dict[str, Any]]:
    """
    Selects a stratified mixture of customer profiles across risk tiers:
    - n_high: Top High risk customers (prob >= 0.44)
    - n_medium: Medium risk customers (0.20 <= prob < 0.44)
    - n_low: Low risk customers (prob < 0.20)
    """
    explainer = ChurnExplainer(model_path)
    probs = explainer.model.predict_proba(X_data)[:, 1]
    
    high_indices = np.where(probs >= DECISION_THRESHOLD)[0]
    med_indices = np.where((probs >= 0.20) & (probs < DECISION_THRESHOLD))[0]
    low_indices = np.where(probs < 0.20)[0]
    
    high_sorted = high_indices[np.argsort(-probs[high_indices])][:n_high]
    med_sorted = med_indices[np.argsort(-probs[med_indices])][:n_medium]
    low_sorted = low_indices[np.argsort(-probs[low_indices])][:n_low]
    
    selected_indices = np.concatenate([high_sorted, med_sorted, low_sorted])
    
    results = []
    for idx in selected_indices:
        row = X_data.iloc[[idx]]
        c_id = str(customer_ids.iloc[idx]) if customer_ids is not None else f"Cust_{idx}"
        profile = get_customer_risk_profile(row, customer_id=c_id, explainer=explainer)
        results.append(profile)
        
    return results

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    
    data_path = 'data/raw/business_churn.csv'
    print(f"Loading test data from {data_path}...")
    raw_df = pd.read_csv(data_path)
    clean_df = load_and_clean_data(data_path)
    
    # Preserve original customer_ids for matching
    ids = raw_df['customer_id'] if 'customer_id' in raw_df.columns else None
    
    X, y = build_features(clean_df)
    
    if ids is not None:
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, ids, test_size=0.20, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        ids_test = None
        
    print(f"\nExecuting get_at_risk_customers(top_n=5) with threshold={DECISION_THRESHOLD}...")
    top_risk_profiles = get_at_risk_customers(X_test, customer_ids=ids_test, top_n=5)
    
    print("\n================ TOP 5 AT-RISK CUSTOMER PROFILES ================")
    print(json.dumps(top_risk_profiles, indent=2))
