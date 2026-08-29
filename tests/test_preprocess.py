import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import load_and_clean_data

def test_load_and_clean_data(tmp_path):
    # Sample test dataset
    raw_data = pd.DataFrame({
        'customerID': ['001', '002'],
        'gender': ['Female', 'Male'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'No'],
        'tenure': [1, 0],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['No', 'No phone service'],
        'InternetService': ['DSL', 'No'],
        'OnlineSecurity': ['Yes', 'No internet service'],
        'OnlineBackup': ['No', 'No internet service'],
        'DeviceProtection': ['No', 'No internet service'],
        'TechSupport': ['Yes', 'No internet service'],
        'StreamingTV': ['No', 'No internet service'],
        'StreamingMovies': ['No', 'No internet service'],
        'Contract': ['Month-to-month', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check'],
        'MonthlyCharges': [29.85, 20.0],
        'TotalCharges': ['29.85', ' '],
        'Churn': ['No', 'Yes']
    })
    
    file_path = tmp_path / "test_raw.csv"
    raw_data.to_csv(file_path, index=False)
    
    df_clean = load_and_clean_data(str(file_path))
    
    # Assert customerID is dropped
    assert 'customerID' not in df_clean.columns
    # Assert TotalCharges is float and blank converted to 0
    assert df_clean['TotalCharges'].dtype in [np.float64, float]
    assert df_clean.loc[1, 'TotalCharges'] == 0.0
    # Assert Churn encoded as 1/0
    assert df_clean.loc[0, 'Churn'] == 0
    assert df_clean.loc[1, 'Churn'] == 1
    # Assert collapsed service categories
    assert df_clean.loc[1, 'TechSupport'] == 'No'
    assert df_clean.loc[1, 'MultipleLines'] == 'No'
