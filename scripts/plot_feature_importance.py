import os
import sys
sys.path.append('.')
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.data.preprocess import load_and_clean_data
from src.features.build_features import build_features

FEATURE_LABELS = {
    "monthly_logins": "Monthly Logins",
    "csat_score": "CSAT Score (1-10)",
    "tenure_months": "Customer Tenure (months)",
    "monthly_fee": "Monthly Subscription Fee ($)",
    "last_login_days_ago": "Days Since Last Login",
    "support_tickets": "Support Ticket Volume",
    "marketing_click_rate": "Marketing Email Click Rate",
    "payment_failures": "Payment Failures Count",
    "escalations": "Technical Escalations",
    "nps_score": "NPS Rating"
}

def generate_shap_bar_chart():
    os.makedirs('docs', exist_ok=True)
    
    # 1. Load data and split exactly as in training/evaluation pipeline
    raw_path = 'data/raw/business_churn.csv'
    clean_df = load_and_clean_data(raw_path)
    X, y = build_features(clean_df)
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 2. Load trained model and compute SHAP in standard log-odds space (matching src/models/explain.py)
    model = joblib.load('models/churn_model.pkl')
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    
    if isinstance(shap_vals, list):
        shap_matrix = shap_vals[1]
    elif len(shap_vals.shape) == 3:
        shap_matrix = shap_vals[:, :, 1]
    else:
        shap_matrix = shap_vals
        
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    
    df_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=True)  # Ascending for horizontal bar chart
    
    top_10 = df_imp.tail(10)
    
    # 3. Create high-resolution Matplotlib chart
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor('#0f172a')  # Dark slate background
    ax.set_facecolor('#1e293b')
    
    labels = [FEATURE_LABELS.get(f, f.replace('_', ' ').title()) for f in top_10['feature']]
    values = top_10['importance'].values
    
    # Plasma gradient for bars
    colors = plt.cm.plasma(np.linspace(0.4, 0.85, len(values)))
    
    bars = ax.barh(labels, values, color=colors, height=0.65, edgecolor='none', zorder=3)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.03, 
            bar.get_y() + bar.get_height()/2, 
            f"{width:.4f}", 
            va='center', 
            ha='left', 
            color='#f8fafc', 
            fontsize=10, 
            fontweight='bold'
        )
        
    ax.set_title("Global Feature Importance (Mean |SHAP Value|)", fontsize=16, fontweight='bold', color='#f8fafc', pad=20)
    ax.set_xlabel("Mean Absolute SHAP Value (Log-Odds Impact on Model Output)", fontsize=11, fontweight='medium', color='#94a3b8', labelpad=10)
    ax.set_xlim(0, max(values) * 1.15)
    
    ax.tick_params(colors='#cbd5e1', labelsize=11)
    ax.grid(color='#334155', linestyle='--', linewidth=0.7, alpha=0.7, zorder=0)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.tight_layout()
    output_path = 'docs/feature_importance.png'
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Successfully saved global SHAP feature importance plot to {output_path}")
    
    # Print top 10 as formatted text
    df_desc = top_10.sort_values('importance', ascending=False)
    print("\n================ TOP 10 GLOBAL FEATURE IMPORTANCES (LOG-ODDS SPACE) ================")
    print(f"{'Rank':<6} {'Feature Name':<28} {'Mean |SHAP Value|':<18}")
    print("-" * 54)
    for rank, (_, row) in enumerate(df_desc.iterrows(), 1):
        feat_clean = FEATURE_LABELS.get(row['feature'], row['feature'])
        print(f"{rank:<6} {feat_clean:<28} {row['importance']:.4f}")
    print("===================================================================\n")

if __name__ == '__main__':
    generate_shap_bar_chart()
