import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocess import load_and_clean_data
from src.features.build_features import build_features
from scripts.run_pipeline import get_at_risk_customers, get_sampled_risk_customers

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
    """Format a clean, human-readable SHAP feature description based on sensible business scales."""
    if feat == 'monthly_logins':
        if val < 5:
            return f"Low Monthly Logins ({val:.0f}/mo)"
        elif val <= 15:
            return f"Moderate Monthly Logins ({val:.0f}/mo)"
        else:
            return f"High Monthly Logins ({val:.0f}/mo)"
    elif feat == 'csat_score':
        if val <= 4:
            return f"Low CSAT Score ({val:.0f}/10)"
        elif val <= 7:
            return f"Average CSAT Score ({val:.0f}/10)"
        else:
            return f"High CSAT Score ({val:.0f}/10)"
    elif feat == 'nps_score':
        if val <= 5:
            return f"Low NPS Score ({val:.0f}/10)"
        elif val <= 8:
            return f"Average NPS Score ({val:.0f}/10)"
        else:
            return f"High NPS Score ({val:.0f}/10)"
    elif feat == 'tenure_months':
        if val < 12:
            return f"Short Customer Tenure ({val:.0f} mos)"
        elif val < 24:
            return f"Moderate Customer Tenure ({val:.0f} mos)"
        else:
            return f"Long Customer Tenure ({val:.0f} mos)"
    elif feat == 'payment_failures':
        return f"Payment Failures ({val:.0f} failed)"
    elif feat == 'support_tickets':
        return f"Frequent Support Tickets ({val:.0f} tickets)"
    elif feat == 'escalations':
        return f"Support Escalations ({val:.0f} escalations)"
    elif feat == 'monthly_fee':
        return f"Monthly Fee (${val:.2f})"
    elif feat.startswith('contract_type_'):
        return f"Contract: {feat.replace('contract_type_', '').title()}"
    elif feat.startswith('complaint_type_'):
        return f"Complaint: {feat.replace('complaint_type_', '').title()}"
    elif feat.startswith('payment_method_'):
        return f"Payment: {feat.replace('payment_method_', '').title()}"
    else:
        clean = FEATURE_NAMES_MAP.get(feat, feat.replace('_', ' ').title())
        return f"{clean} ({val})"

from src.agent.risk_explainer import explain_risk
from src.agent.retention_actions import recommend_actions

def is_genuinely_positive_factor(feat: str, val: float) -> bool:
    """Check if raw value of feature is genuinely positive/protective."""
    if feat == 'csat_score' and val < 5:
        return False
    if feat == 'nps_score' and val < 6:
        return False
    if feat == 'monthly_logins' and val < 5:
        return False
    if feat == 'payment_failures' and val >= 1:
        return False
    if feat == 'escalations' and val >= 1:
        return False
    return True

def generate_customer_text_snippets(raw_row: pd.Series, top_factors: list, risk_level: str) -> tuple:
    """
    Generates realistic support ticket excerpt and customer feedback snippet
    consistent with customer's actual risk level and raw metrics.
    """
    csat = raw_row.get('csat_score', 7) if raw_row is not None and 'csat_score' in raw_row else 7
    tickets = raw_row.get('support_tickets', 0) if raw_row is not None and 'support_tickets' in raw_row else 0
    failures = raw_row.get('payment_failures', 0) if raw_row is not None and 'payment_failures' in raw_row else 0
    logins = raw_row.get('monthly_logins', 10) if raw_row is not None and 'monthly_logins' in raw_row else 10
    escalations = raw_row.get('escalations', 0) if raw_row is not None and 'escalations' in raw_row else 0
    tenure = raw_row.get('tenure_months', 12) if raw_row is not None and 'tenure_months' in raw_row else 12
    days_ago = raw_row.get('last_login_days_ago', 5) if raw_row is not None and 'last_login_days_ago' in raw_row else 5
    
    top_feat = top_factors[0]['feature'] if top_factors else ""

    if risk_level == 'Low':
        # Genuinely positive / neutral snippets for Low risk customers
        if logins >= 20:
            ticket = f"Feature Request #{8000 + int(tenure*3)}: Requested custom CSV report scheduling feature for team leads."
            feedback = "Customer Review: 'Fantastic platform and super responsive support; our team relies on it daily!'"
        elif csat >= 7:
            ticket = f"General Question #{8100 + int(tenure*2)}: Inquired about setting up SSO authentication for new team members."
            feedback = f"CSAT Survey ({int(csat)}/10): 'Seamless onboarding experience and immediate value delivered to our team.'"
        else:
            ticket = f"Account Inquiry #{8200 + int(tenure*2)}: Requested assistance adding 2 additional user seats for the upcoming quarter."
            feedback = "Quarterly Review: 'Product is performing well and support team answered our questions promptly.'"

    elif risk_level == 'Medium':
        # Mixed / moderate snippets for Medium risk customers
        if csat <= 4:
            ticket = f"Support Ticket #{3100 + int(tenure*2)}: Inquired about optimization tips for dashboard load times during peak hours."
            feedback = f"CSAT Survey ({int(csat)}/10): 'Core functionality is good, but would appreciate faster support turnaround on minor issues.'"
        elif failures >= 1:
            ticket = f"Billing Note #{1100 + int(tenure*3)}: Requested updated invoice copy after card update."
            feedback = f"Feedback Note: 'Overall satisfied with the service, resolving billing detail update.'"
        else:
            ticket = f"General Inquiry #{7000 + int(tenure*2)}: User asked for clarification on monthly API call rate limits and tier caps."
            feedback = f"Quarterly Feedback: 'Evaluating feature usage prior to upcoming contract renewal discussion.'"

    else:
        # Genuinely negative / urgent snippets for High risk customers
        if failures >= 2 or top_feat == 'payment_failures':
            ticket = f"Billing Escalation #{1000 + int(tenure*3)}: Recurring payment failed twice during renewal; user requested payment link update."
            feedback = f"CSAT Survey ({int(csat)}/10): 'Payment failures causing account access blocks; urgent resolution required.'"
        elif escalations >= 1 or top_feat == 'escalations':
            ticket = f"Escalated Ticket #{2000 + int(tenure*2)}: Critical integration bug reopened twice; requested executive callback."
            feedback = f"CSAT Survey ({int(csat)}/10): 'Unresolved bugs impacting daily operations; considering alternative solutions.'"
        elif tickets >= 3 or top_feat == 'support_tickets':
            ticket = f"Support Ticket #{3000 + int(tenure*4)}: Third ticket this month regarding reporting export failures and dashboard lag."
            feedback = f"CSAT Survey ({int(csat)}/10): 'Frequent system downtime and slow ticket responses are severely impacting our team.'"
        elif logins < 5 or days_ago > 20 or top_feat == 'monthly_logins':
            ticket = f"Account Warning #{4000 + int(tenure*5)}: Admin requested seat reassignment guide after team inactivity warning."
            feedback = f"Account Feedback: 'Team usage has dropped due to lack of training; evaluating subscription downsizing.'"
        else:
            ticket = f"Service Ticket #{5000 + int(tenure*2)}: Reported onboarding roadblock and requested cancellation terms."
            feedback = f"CSAT Survey ({int(csat)}/10): 'Dissatisfied with platform adoption and support responsiveness.'"

    return ticket, feedback

def transform_profile_for_v2(profile: dict, raw_row: pd.Series = None) -> dict:
    """
    Transform a raw pipeline risk profile into the v2 dashboard shape
    including realistic support ticket and customer feedback snippets.
    """
    top_factors = profile.get("top_risk_factors", [])
    risk_level = profile.get("risk_level", "Medium")
    
    ticket_excerpt, feedback_snippet = generate_customer_text_snippets(raw_row, top_factors, risk_level)
    
    # Pass snippets into profile and re-explain
    profile["support_ticket_excerpt"] = ticket_excerpt
    profile["feedback_snippet"] = feedback_snippet
    explanation = explain_risk(profile)
    
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
            # Only include under "decreasing risk" if feature is genuinely positive/protective
            if is_genuinely_positive_factor(feat, val):
                feat_desc = format_feature_name(feat, val, is_increasing=False)
                decreasing.append({"feature": feat_desc, "value": abs_shap})
            
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
            
    actions = recommend_actions(explanation, top_risk_factors=top_factors, risk_level=risk_level)
    
    return {
        "customer_id": profile.get("customer_id", "N/A"),
        "churn_probability": round(profile.get("churn_probability", 0.0), 4),
        "risk_level": risk_level,
        "metrics": metrics,
        "support_ticket_excerpt": ticket_excerpt,
        "feedback_snippet": feedback_snippet,
        "explanation": explanation,
        "shap": {
            "increasing": increasing,
            "decreasing": decreasing
        },
        "recommended_actions": actions
    }

def generate_dashboard_data(output_path: str = 'data/processed/dashboard_data.json', n_high: int = 10, n_medium: int = 5, n_low: int = 3):
    """
    Generates customer risk profiles across a realistic risk mix:
    - 10 High risk customers (prob >= 0.44)
    - 5 Medium risk customers (0.20 <= prob < 0.44)
    - 3 Low risk customers (prob < 0.20)
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
        
    print(f"Generating risk profiles for stratified mix: {n_high} High / {n_medium} Medium / {n_low} Low risk customers...")
    raw_profiles = get_sampled_risk_customers(
        X_test, 
        customer_ids=ids_test, 
        n_high=n_high, 
        n_medium=n_medium, 
        n_low=n_low
    )
    
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
