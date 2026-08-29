import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List

# Production Decision Threshold (Option B: Recall >= 70%)
DECISION_THRESHOLD = 0.44

class ChurnPredictor:
    def __init__(self, model_path: str = 'models/churn_model.pkl', threshold: float = DECISION_THRESHOLD):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def get_risk_level(self, probability: float) -> str:
        """
        Categorizes churn probability into High, Medium, or Low risk tiers based on production 0.44 threshold:
        - High: prob >= 0.44 (Classified as Churn)
        - Medium: 0.20 <= prob < 0.44
        - Low: prob < 0.20
        """
        if probability >= self.threshold:
            return "High"
        elif probability >= 0.20:
            return "Medium"
        else:
            return "Low"

    def predict_single(self, customer_row: Union[pd.DataFrame, pd.Series]) -> Dict[str, Any]:
        """
        Predict churn probability, binary label (0/1), and risk level for a single customer row.
        """
        if isinstance(customer_row, pd.Series):
            customer_row = customer_row.to_frame().T
            
        prob = float(self.model.predict_proba(customer_row)[0, 1])
        prediction = 1 if prob >= self.threshold else 0
        risk_level = self.get_risk_level(prob)
        
        return {
            "churn_probability": round(prob, 4),
            "churn_prediction": prediction,
            "risk_level": risk_level,
            "threshold_used": self.threshold
        }

    def predict_batch(self, X_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict churn probabilities, labels, and risk levels for a DataFrame batch.
        """
        probs = self.model.predict_proba(X_data)[:, 1]
        preds = (probs >= self.threshold).astype(int)
        risk_levels = [self.get_risk_level(p) for p in probs]
        
        res_df = X_data.copy()
        res_df['churn_probability'] = round(pd.Series(probs), 4)
        res_df['churn_prediction'] = preds
        res_df['risk_level'] = risk_levels
        return res_df

def predict_churn(customer_row: pd.DataFrame, threshold: float = DECISION_THRESHOLD, model_path: str = 'models/churn_model.pkl') -> Dict[str, Any]:
    predictor = ChurnPredictor(model_path=model_path, threshold=threshold)
    return predictor.predict_single(customer_row)
