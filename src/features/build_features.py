import pandas as pd
from typing import Tuple

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target vector y from clean business churn DataFrame:
    - Extract target column 'churn' as y
    - One-hot encode remaining categorical columns (contract_type, signup_channel, complaint_type, payment_method, customer_segment, survey_response)
    - Preserve numerical features as-is
    - Return (X, y)
    """
    if "churn" in df.columns:
        y = df["churn"].copy()
        X_raw = df.drop(columns=["churn"])
    else:
        y = None
        X_raw = df.copy()
        
    # Identify categorical columns to one-hot encode
    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # One-hot encode categoricals with drop_first=True
    if cat_cols:
        X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=float)
    else:
        X = X_raw.copy()
        
    return X, y
