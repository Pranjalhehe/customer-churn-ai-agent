import os
import sys
import io
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import load_and_clean_data
from src.features.build_features import build_features
from src.models.explain import ChurnExplainer
from src.models.predict import ChurnPredictor
from src.agent.risk_explainer import explain_risk, translate_feature_name
from src.agent.retention_actions import recommend_actions
from scripts.generate_dashboard_data import generate_customer_text_snippets

REQUIRED_RAW_COLUMNS = [
    'age', 'tenure_months', 'monthly_logins', 'weekly_active_days', 
    'avg_session_time', 'features_used', 'usage_growth_rate', 
    'last_login_days_ago', 'monthly_fee', 'total_revenue', 
    'payment_failures', 'discount_applied', 'price_increase_last_3m', 
    'support_tickets', 'avg_resolution_time', 'csat_score', 
    'escalations', 'email_open_rate', 'marketing_click_rate', 
    'nps_score', 'referral_count', 'customer_segment', 
    'signup_channel', 'contract_type', 'payment_method', 
    'complaint_type', 'survey_response'
]

FEATURE_DRIVER_MAP = {
    'csat_score': 'low CSAT scores',
    'tenure_months': 'short customer tenure',
    'monthly_logins': 'low monthly logins',
    'support_tickets': 'frequent support tickets',
    'payment_failures': 'payment failures',
    'escalations': 'support escalations',
    'monthly_fee': 'high monthly fee',
    'price_increase_last_3m': 'recent price increases',
    'nps_score': 'low NPS scores',
    'weekly_active_days': 'low weekly active days',
    'last_login_days_ago': 'customer inactivity',
    'avg_session_time': 'low session duration',
    'usage_growth_rate': 'declining usage'
}

def validate_csv_schema(df_raw: pd.DataFrame, expected_features: list) -> tuple:
    """
    Validates that the uploaded CSV contains all required raw columns.
    Returns (is_valid, missing_columns_list).
    """
    missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in df_raw.columns]
    if missing_cols:
        return False, missing_cols
    return True, []

def process_uploaded_csv(uploaded_file):
    """
    Full processing pipeline for an uploaded CSV:
    1. Read CSV & Validate schema
    2. Cap at 10 rows
    3. Run load_and_clean_data & build_features
    4. Align features with model.feature_names_in_
    5. Run SHAP, AI Risk Explainer, and Retention Recommendations per row
    6. Return list of customer profiles and summary metrics
    """
    in_streamlit = st.runtime.exists()
    
    # Read raw CSV
    bytes_data = uploaded_file.getvalue()
    df_raw = pd.read_csv(io.BytesIO(bytes_data))
    
    # Load model & get expected features
    model_path = 'models/churn_model.pkl'
    explainer = ChurnExplainer(model_path)
    predictor = ChurnPredictor(model_path)
    expected_features = list(explainer.model.feature_names_in_)
    
    # Schema validation
    is_valid, missing_cols = validate_csv_schema(df_raw, expected_features)
    if not is_valid:
        if in_streamlit:
            st.error(f"❌ **Schema Validation Failed!** Uploaded CSV is missing {len(missing_cols)} expected column(s):\n\n" +
                     f"**Missing Columns:** `{', '.join(missing_cols)}`\n\n" +
                     "Please upload a valid CSV file containing all required customer metrics.")
            with st.expander("📋 View Full Expected Raw Columns"):
                st.write(REQUIRED_RAW_COLUMNS)
            with st.expander("🤖 View Exact Model Features (post-preprocessing)"):
                st.write(expected_features)
        else:
            print(f"Schema Validation Failed! Missing columns: {missing_cols}")
        return None, None
        
    # Cap processing at first 10 rows
    total_raw_rows = len(df_raw)
    if total_raw_rows > 10:
        if in_streamlit:
            st.info(f"ℹ️ Uploaded CSV contains {total_raw_rows} rows. **Processing first 10 customers for this demo.**")
        else:
            print(f"Uploaded CSV contains {total_raw_rows} rows. Processing first 10 customers for this demo.")
        df_raw_capped = df_raw.iloc[:10].copy()
    else:
        df_raw_capped = df_raw.copy()
        
    # Preprocess & build feature matrix
    clean_df = load_and_clean_data(io.BytesIO(bytes_data))
    clean_df_capped = clean_df.iloc[:len(df_raw_capped)].copy()
    
    X_raw, _ = build_features(clean_df_capped)
    
    # Align X columns with exact model expected features
    X = X_raw.reindex(columns=expected_features, fill_value=0)
    
    # Progress UI
    progress_bar = st.progress(0.0) if in_streamlit else None
    status_text = st.empty() if in_streamlit else None
    
    results = []
    top_factor_counts = {}
    
    total_to_process = len(X)
    for i in range(total_to_process):
        cid = str(df_raw_capped.iloc[i]['customer_id']) if 'customer_id' in df_raw_capped.columns else f"DEMO_{i+1:05d}"
        if status_text:
            status_text.markdown(f"⏳ **Processing customer {i+1} of {total_to_process}:** `{cid}`...")
        
        row_X = X.iloc[[i]]
        row_raw = df_raw_capped.iloc[i]
        
        # 1. SHAP & Probability
        shap_output = explainer.get_top_risk_factors(row_X, top_n=3)
        prob = shap_output["churn_probability"]
        risk_lvl = predictor.get_risk_level(prob)
        top_factors = shap_output.get("top_risk_factors", [])
        
        # 2. Text Snippets & AI Risk Explainer
        ticket_excerpt, feedback_snippet = generate_customer_text_snippets(row_raw, top_factors, risk_lvl)
        shap_output["support_ticket_excerpt"] = ticket_excerpt
        shap_output["feedback_snippet"] = feedback_snippet
        
        explanation = explain_risk(shap_output)
        
        # 3. Retention Recommendations
        actions = recommend_actions(explanation, top_risk_factors=top_factors, risk_level=risk_lvl)
        
        # Track primary risk factor
        if top_factors:
            p_feat = top_factors[0]['feature']
            top_factor_counts[p_feat] = top_factor_counts.get(p_feat, 0) + 1
            
        results.append({
            "customer_id": cid,
            "churn_probability": prob,
            "risk_level": risk_lvl,
            "top_risk_factors": top_factors,
            "primary_factor": top_factors[0]['feature'] if top_factors else "N/A",
            "support_ticket_excerpt": ticket_excerpt,
            "feedback_snippet": feedback_snippet,
            "explanation": explanation,
            "recommended_actions": actions
        })
        
        if progress_bar:
            progress_bar.progress((i + 1) / total_to_process)
        
    if status_text:
        status_text.success(f"✅ Real-time processing complete for all {total_to_process} customers!")
    
    # Calculate Summary Stats
    total_processed = len(results)
    high_count = sum(1 for r in results if r['risk_level'] == 'High')
    med_count = sum(1 for r in results if r['risk_level'] == 'Medium')
    low_count = sum(1 for r in results if r['risk_level'] == 'Low')
    
    most_common_feat = max(top_factor_counts, key=top_factor_counts.get) if top_factor_counts else "general risk"
    driver_desc = FEATURE_DRIVER_MAP.get(most_common_feat, most_common_feat.replace('_', ' '))
    
    if high_count > 0:
        summary_phrase = f"{high_count} of {total_processed} customers are High risk, primarily driven by {driver_desc}"
    else:
        summary_phrase = f"{total_processed} customers processed ({med_count} Medium, {low_count} Low), primarily driven by {driver_desc}"
        
    summary_stats = {
        "total_processed": total_processed,
        "high_count": high_count,
        "med_count": med_count,
        "low_count": low_count,
        "most_common_factor": most_common_feat,
        "summary_phrase": summary_phrase
    }
    
    return results, summary_stats

def main():
    st.set_page_config(page_title="Real-Time Customer Churn Predictor", page_icon="⚡", layout="wide")
    
    st.title("⚡ Real-Time Customer Churn Batch Predictor")
    st.caption("Upload customer data CSV to run real-time churn predictions, SHAP explanations, and AI retention strategies.")
    
    st.markdown("---")
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Customer Batch CSV", 
        type=["csv"], 
        help="Upload a CSV matching customer dataset schema (e.g. demo_samples/fresh_customers.csv)"
    )
    
    if uploaded_file is not None:
        st.write(f"📁 **File Uploaded:** `{uploaded_file.name}` ({uploaded_file.size} bytes)")
        
        # Process CSV
        results, summary_stats = process_uploaded_csv(uploaded_file)
        
        if results and summary_stats:
            st.markdown("---")
            st.header("📊 Executive Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Processed", summary_stats["total_processed"])
            col2.metric("High Risk (🚨)", summary_stats["high_count"])
            col3.metric("Medium Risk (⚠️)", summary_stats["med_count"])
            col4.metric("Low Risk (✅)", summary_stats["low_count"])
            
            # Summary banner callout
            st.info(f"💡 **Key Synthesis:** {summary_stats['summary_phrase']}.")
            
            st.markdown("---")
            st.header("📋 Results Overview Table")
            
            # Overview Table
            table_data = []
            for r in results:
                p_feat = r["primary_factor"]
                clean_p_feat = FEATURE_DRIVER_MAP.get(p_feat, p_feat.replace("_", " ").title())
                table_data.append({
                    "Customer ID": r["customer_id"],
                    "Churn Probability": f"{r['churn_probability']:.1%}",
                    "Risk Level": r["risk_level"],
                    "Primary Risk Driver": clean_p_feat
                })
            table_df = pd.DataFrame(table_data)
            st.dataframe(table_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.header("🔍 Customer Risk Details & AI Retention Strategies")
            st.caption("Click any customer below to expand SHAP factors, natural language explanation, and recommended retention actions.")
            
            for r in results:
                risk_icon = "🔴" if r["risk_level"] == "High" else ("🟡" if r["risk_level"] == "Medium" else "🟢")
                expander_label = f"{risk_icon} {r['customer_id']} — Risk: {r['churn_probability']:.1%} ({r['risk_level']} Risk)"
                
                with st.expander(expander_label):
                    c1, c2 = st.columns([1, 1])
                    
                    with c1:
                        st.markdown("### 📊 Metrics & SHAP Factors")
                        st.write(f"**Customer ID:** `{r['customer_id']}`")
                        st.write(f"**Churn Risk:** `{r['churn_probability']:.1%}` ({r['risk_level']})")
                        
                        st.write("**Top SHAP Risk Factors:**")
                        for factor in r["top_risk_factors"]:
                            translated = translate_feature_name(factor["feature"], factor["value"], factor["direction"])
                            direction_icon = "🔺" if factor["direction"] == "increased risk" else "🔻"
                            st.write(f"{direction_icon} **{translated}** (SHAP impact: `{factor['shap_value']:+.4f}`)")
                            
                        if r["support_ticket_excerpt"]:
                            st.write(f"**Recent Support Ticket:** _{r['support_ticket_excerpt']}_")
                        if r["feedback_snippet"]:
                            st.write(f"**Customer Feedback:** _{r['feedback_snippet']}_")
                            
                    with c2:
                        st.markdown("### 🤖 AI Risk Explanation")
                        st.info(r["explanation"])
                        
                        st.markdown("### 🎯 Recommended Retention Actions")
                        for idx, action in enumerate(r["recommended_actions"], 1):
                            st.write(f"**{idx}.** {action}")

if __name__ == '__main__':
    main()
