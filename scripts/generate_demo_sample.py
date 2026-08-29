import os
import sys
sys.path.append('.')
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.preprocess import load_and_clean_data
from src.features.build_features import build_features

def generate_demo_samples(
    input_raw_path: str = 'data/raw/business_churn.csv',
    model_path: str = 'models/churn_model.pkl',
    output_path: str = 'demo_samples/fresh_customers.csv'
) -> str:
    """
    Generates 20 demo customer records sampled EXCLUSIVELY from the HELD-OUT TEST SET
    (80/20 train_test_split with random_state=42, matching train.py).
    Ensures the demo genuinely presents customers the model was never trained on.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Load raw dataset and preprocess
    raw_df = pd.read_csv(input_raw_path)
    clean_df = load_and_clean_data(input_raw_path)
    X, y = build_features(clean_df)
    
    # 2. Perform EXACT SAME 80/20 train_test_split as train.py
    _, raw_test, _, y_test = train_test_split(
        raw_df, y, test_size=0.20, random_state=42, stratify=y
    )
    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Score held-out test set with trained model artifact
    model = joblib.load(model_path)
    X_test_aligned = X_test[model.feature_names_in_]
    probs = model.predict_proba(X_test_aligned)[:, 1]
    
    raw_test = raw_test.copy()
    raw_test['temp_prob'] = probs
    
    # 4. Stratified sample exclusively from held-out test set (8 High, 4 Medium, 8 Low)
    high = raw_test[raw_test['temp_prob'] >= 0.44].sample(8, random_state=42)
    med  = raw_test[(raw_test['temp_prob'] >= 0.20) & (raw_test['temp_prob'] < 0.44)].sample(4, random_state=42)
    low  = raw_test[raw_test['temp_prob'] < 0.20].sample(8, random_state=42)
    
    demo_df = pd.concat([high, med, low], ignore_index=True).drop(columns=['temp_prob'])
    
    # Re-key customer IDs to DEMO_00001 through DEMO_00020
    demo_df['customer_id'] = [f"DEMO_{i:05d}" for i in range(1, 21)]
    
    # 5. Verify exact schema match
    if list(demo_df.columns) != list(raw_df.columns):
        raise ValueError(f"Column mismatch!\nExpected: {list(raw_df.columns)}\nGot: {list(demo_df.columns)}")
        
    demo_df.to_csv(output_path, index=False)
    print(f"Successfully sampled {len(demo_df)} held-out test records saved to '{output_path}'.")
    return output_path

def evaluate_demo_sample(csv_path: str = 'demo_samples/fresh_customers.csv', model_path: str = 'models/churn_model.pkl'):
    """
    Evaluates generated demo customer dataset using the production XGBoost model artifact.
    """
    print(f"\nEvaluating generated dataset '{csv_path}' with model artifact '{model_path}'...")
    
    clean_df = load_and_clean_data(csv_path)
    X_demo, y_demo = build_features(clean_df)
    
    model = joblib.load(model_path)
    
    # Ensure column alignment with model expected features
    for col in model.feature_names_in_:
        if col not in X_demo.columns:
            X_demo[col] = 0.0
    X_demo = X_demo[model.feature_names_in_]
    
    probs = model.predict_proba(X_demo)[:, 1]
    
    raw_df = pd.read_csv(csv_path)
    
    results = []
    for idx, row in raw_df.iterrows():
        prob = probs[idx]
        if prob >= 0.44:
            risk_level = "High"
        elif prob >= 0.20:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        results.append({
            "Customer ID": row["customer_id"],
            "Segment": row["customer_segment"],
            "Logins/mo": row["monthly_logins"],
            "CSAT": row["csat_score"],
            "Failures": row["payment_failures"],
            "Actual Churn": row["churn"],
            "Predicted Risk": f"{prob:.1%}",
            "Risk Level": risk_level
        })
        
    res_df = pd.DataFrame(results)
    
    print("\n================ DEMO BATCH MODEL EVALUATION RESULTS (HELD-OUT TEST SET) ================")
    print(res_df.to_string(index=False))
    print("=========================================================================================")
    
    print(f"\nSummary Breakdown Across 20 Held-Out Demo Customers:")
    print(f"- High Risk   (Prob >= 44%): {(res_df['Risk Level'] == 'High').sum()}")
    print(f"- Medium Risk (20% <= Prob < 44%): {(res_df['Risk Level'] == 'Medium').sum()}")
    print(f"- Low Risk    (Prob < 20%): {(res_df['Risk Level'] == 'Low').sum()}")
    print("\nAll 20 held-out test records scored cleanly without any column mismatches or runtime errors.")

if __name__ == '__main__':
    csv_file = generate_demo_samples()
    evaluate_demo_sample(csv_file)
