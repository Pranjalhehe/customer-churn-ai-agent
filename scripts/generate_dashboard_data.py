import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocess import load_and_clean_data
from src.features.build_features import build_features
from scripts.run_pipeline import get_at_risk_customers

FEATURE_NAMES_MAP = {
    'monthly_logins': 'Monthly Logins',
    'csat_score': 'CSAT Score',
    'tenure_months': 'Customer Tenure',
    'support_tickets': 'Support Tickets',
    'payment_failures': 'Payment Failures',
    'escalations': 'Support Escalations',
    'monthly_fee': 'Monthly Fee',
    'nps_score': 'NPS Score',
    'weekly_active_days': 'Weekly Active Days',
    'avg_session_time': 'Avg Session Time',
    'last_login_days_ago': 'Days Since Last Login',
    'usage_growth_rate': 'Usage Growth Rate',
    'total_revenue': 'Total Revenue'
}

def format_feature_name(feat: str, val: float, is_increasing: bool) -> str:
    """Format a clean, human-readable SHAP feature description."""
    if feat == 'monthly_logins':
        return f"{'Low' if val < 5 else 'High'} Monthly Logins ({val:.0f}/mo)"
    elif feat == 'csat_score':
        return f"{'Low' if val <= 4 else 'High'} CSAT Score ({val:.0f}/10)"
    elif feat == 'tenure_months':
        return f"{'Short' if val < 12 else 'Long'} Customer Tenure ({val:.0f} mos)"
    elif feat == 'payment_failures':
        return f"Payment Failures ({val:.0f} failed)"
    elif feat == 'support_tickets':
        return f"Frequent Support Tickets ({val:.0f} tickets)"
    elif feat == 'escalations':
        return f"Support Escalations ({val:.0f} escalations)"
    elif feat == 'monthly_fee':
        return f"Monthly Fee (${val:.2f})"
    elif feat == 'nps_score':
        return f"NPS Score ({val:.0f}/10)"
    elif feat.startswith('contract_type_'):
        return f"Contract: {feat.replace('contract_type_', '').title()}"
    elif feat.startswith('complaint_type_'):
        return f"Complaint: {feat.replace('complaint_type_', '').title()}"
    elif feat.startswith('payment_method_'):
        return f"Payment: {feat.replace('payment_method_', '').title()}"
    else:
        clean = FEATURE_NAMES_MAP.get(feat, feat.replace('_', ' ').title())
        return f"{clean} ({val})"

def transform_profile_for_v2(profile: dict, raw_row: pd.Series = None) -> dict:
    """
    Transform a raw pipeline risk profile into the v2 dashboard shape:
    {
      customer_id, churn_probability, risk_level,
      metrics: [{label, value}, ...],
      shap: {
        increasing: [{feature, value}, ...],
        decreasing: [{feature, value}, ...]
      },
      recommended_actions: [...]
    }
    """
    top_factors = profile.get("top_risk_factors", [])
    
    increasing = []
    decreasing = []
    
    for factor in top_factors:
        feat = factor.get("feature", "")
        val = factor.get("value", 0)
        shap_val = factor.get("shap_value", 0)
        direction = factor.get("direction", "")
        
        abs_shap = round(abs(shap_val), 4)
        
        if direction == "increased risk" or shap_val > 0:
            feat_desc = format_feature_name(feat, val, is_increasing=True)
            increasing.append({"feature": feat_desc, "value": abs_shap})
        else:
            feat_desc = format_feature_name(feat, val, is_increasing=False)
            decreasing.append({"feature": feat_desc, "value": abs_shap})
            
    # Build metrics array from raw customer row if available
    metrics = []
    if raw_row is not None:
        if "tenure_months" in raw_row:
            metrics.append({"label": "Tenure", "value": f"{int(raw_row['tenure_months'])} months"})
        if "monthly_logins" in raw_row:
            metrics.append({"label": "Monthly Logins", "value": f"{int(raw_row['monthly_logins'])} / mo"})
        if "csat_score" in raw_row:
            metrics.append({"label": "CSAT Score", "value": f"{int(raw_row['csat_score'])} / 10"})
        if "monthly_fee" in raw_row:
            metrics.append({"label": "Monthly Fee", "value": f"${float(raw_row['monthly_fee']):.2f}"})
        elif "contract_type" in raw_row:
            metrics.append({"label": "Contract", "value": str(raw_row['contract_type'])})
    else:
        for f in top_factors[:4]:
            feat = f.get("feature", "")
            val = f.get("value", 0)
            clean_lbl = FEATURE_NAMES_MAP.get(feat, feat.replace("_", " ").title())
            metrics.append({"label": clean_lbl, "value": str(val)})
            
    return {
        "customer_id": profile.get("customer_id", "N/A"),
        "churn_probability": round(profile.get("churn_probability", 0.0), 4),
        "risk_level": profile.get("risk_level", "Medium"),
        "metrics": metrics,
        "shap": {
            "increasing": increasing,
            "decreasing": decreasing
        },
        "recommended_actions": profile.get("recommended_actions", [])
    }

def generate_dashboard_data(output_path: str = 'data/processed/dashboard_data.json', top_n: int = 15):
    """
    Generates full customer risk profiles for top N at-risk test customers,
    reshapes them for v2 dashboard design, and saves to JSON.
    """
    raw_path = 'data/raw/business_churn.csv'
    print(f"Loading raw data from {raw_path}...")
    raw_df = pd.read_csv(raw_path)
    clean_df = load_and_clean_data(raw_path)
    
    customer_ids = raw_df['customer_id'] if 'customer_id' in raw_df.columns else None
    X, y = build_features(clean_df)
    
    if customer_ids is not None:
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, customer_ids, test_size=0.20, random_state=42, stratify=y
        )
        _, raw_test = train_test_split(raw_df, test_size=0.20, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        raw_test = None
        ids_test = None
        
    print(f"Generating full risk profiles for top {top_n} at-risk customers...")
    raw_profiles = get_at_risk_customers(X_test, customer_ids=ids_test, top_n=top_n)
    
    v2_profiles = []
    for profile in raw_profiles:
        cid = profile.get("customer_id")
        raw_row = None
        if raw_test is not None and cid is not None and 'customer_id' in raw_test.columns:
            matched = raw_test[raw_test['customer_id'] == cid]
            if not matched.empty:
                raw_row = matched.iloc[0]
                
        v2_profile = transform_profile_for_v2(profile, raw_row=raw_row)
        v2_profiles.append(v2_profile)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(v2_profiles, f, indent=2)
        
    print(f"Successfully saved {len(v2_profiles)} reshaped customer profiles to {output_path}\n")
    return v2_profiles

if __name__ == '__main__':
    data = generate_dashboard_data()
    print("=== FIRST 2 CUSTOMERS FROM GENERATED DASHBOARD DATA (V2 SHAPE) ===")
    print(json.dumps(data[:2], indent=2))
