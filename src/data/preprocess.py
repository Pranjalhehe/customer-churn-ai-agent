import pandas as pd

def load_and_clean_data(path: str = "data/raw/business_churn.csv") -> pd.DataFrame:
    """
    Load raw business churn data and perform initial data cleaning:
    - Load CSV from path
    - Drop customer_id (non-predictive identifier)
    - Drop gender, country, city (demographic features showing negligible correlation with churn)
    - Fill missing values in complaint_type with 'None'
    - Convert binary 'Yes'/'No' flags (discount_applied, price_increase_last_3m) to 1/0 numeric
    - Ensure target 'churn' is numeric 0/1
    """
    df = pd.read_csv(path)
    
    # Drop non-predictive identifier and low-signal demographic columns
    drop_cols = ["customer_id", "gender", "country", "city"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
    
    # Handle missing values in complaint_type
    if "complaint_type" in df.columns:
        df["complaint_type"] = df["complaint_type"].fillna("None")
        
    # Convert Yes/No flags to 1/0 numeric
    if "discount_applied" in df.columns:
        df["discount_applied"] = df["discount_applied"].astype(str).str.strip().map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        
    if "price_increase_last_3m" in df.columns:
        df["price_increase_last_3m"] = df["price_increase_last_3m"].astype(str).str.strip().map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        
    # Target encoding
    if "churn" in df.columns:
        df["churn"] = df["churn"].astype(int)
        
    return df
