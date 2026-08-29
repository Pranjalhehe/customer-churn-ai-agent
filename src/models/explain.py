import os
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, Optional

class ChurnExplainer:
    def __init__(self, model_path: str = 'models/churn_model.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)

    def get_top_risk_factors(self, customer_row: pd.DataFrame, top_n: int = 3) -> Dict[str, Any]:
        """
        Computes SHAP values and returns churn probability + top N risk factors for a customer.
        customer_row: single-row pandas DataFrame matching model features.
        """
        if isinstance(customer_row, pd.Series):
            customer_row = customer_row.to_frame().T
            
        # 1. Churn probability
        churn_prob = float(self.model.predict_proba(customer_row)[0, 1])
        
        # 2. Compute SHAP values
        shap_vals = self.explainer(customer_row).values
        if len(shap_vals.shape) == 3:  # Binary classification with 2 output dimensions
            shap_vector = shap_vals[0, :, 1]
        elif len(shap_vals.shape) == 2:
            shap_vector = shap_vals[0, :]
        else:
            shap_vector = shap_vals
            
        feature_names = customer_row.columns.tolist()
        feature_values = customer_row.iloc[0].values
        
        # Combine into list of dicts
        factors = []
        for feat, val, s_val in zip(feature_names, feature_values, shap_vector):
            factors.append({
                "feature": feat,
                "value": float(val) if isinstance(val, (np.integer, np.floating, int, float)) else val,
                "shap_value": round(float(s_val), 4),
                "abs_shap": abs(float(s_val)),
                "direction": "increased risk" if s_val > 0 else "decreased risk"
            })
            
        # Sort by absolute SHAP value descending
        factors.sort(key=lambda x: x["abs_shap"], reverse=True)
        
        # Format top N risk factors
        top_factors = []
        for item in factors[:top_n]:
            top_factors.append({
                "feature": item["feature"],
                "value": item["value"],
                "shap_value": item["shap_value"],
                "direction": item["direction"]
            })
            
        return {
            "churn_probability": round(churn_prob, 4),
            "top_risk_factors": top_factors
        }

# Global singleton helper function
_explainer_instance: Optional[ChurnExplainer] = None

def get_top_risk_factors(customer_row: pd.DataFrame, top_n: int = 3, model_path: str = 'models/churn_model.pkl') -> Dict[str, Any]:
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = ChurnExplainer(model_path)
    return _explainer_instance.get_top_risk_factors(customer_row, top_n=top_n)

if __name__ == '__main__':
    from src.data.preprocess import load_and_clean_data
    from src.features.build_features import build_features
    from sklearn.model_selection import train_test_split
    import json

    df_clean = load_and_clean_data('data/raw/business_churn.csv')
    X, y = build_features(df_clean)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    explainer = ChurnExplainer('models/churn_model.pkl')
    probs = explainer.model.predict_proba(X_test)[:, 1]
    
    # Pick sample customers: low-risk (< 0.2), high-risk (> 0.8), borderline (0.45 - 0.55)
    low_idx = np.argmin(probs)
    high_idx = np.argmax(probs)
    borderline_idx = np.argmin(np.abs(probs - 0.50))
    
    samples = [
        ("Low Risk Customer", low_idx),
        ("High Risk Customer", high_idx),
        ("Borderline Customer", borderline_idx)
    ]
    
    print("\n================ SHAP EXPLAINER TEST RESULTS ================")
    for label, idx in samples:
        row = X_test.iloc[[idx]]
        res = explainer.get_top_risk_factors(row, top_n=3)
        print(f"\n--- {label} (Actual Churn: {y_test.iloc[idx]}) ---")
        print(json.dumps(res, indent=2))
