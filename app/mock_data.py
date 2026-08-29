"""
Mock customer data. Shape matches what the model pipeline (teammate's side)
is expected to output. Swap MOCK_CUSTOMERS for real model output once it's ready —
nothing else in the app needs to change if the shape matches.
"""

MOCK_CUSTOMERS = [
    {
        "customer_id": "C-1001",
        "name": "Alicia Reyes",
        "churn_probability": 0.82,
        "top_risk_factors": [
            {"feature": "Contract type", "value": "Month-to-month", "impact": 0.28},
            {"feature": "Tech support tickets (90d)", "value": 4, "impact": 0.21},
            {"feature": "Tenure", "value": "3 months", "impact": 0.17},
        ],
        "ticket_text": "Called twice this week about slow speeds, still not fixed. Asked if there's a cheaper plan.",
        "ticket_sentiment": "negative",
        "contract_type": "Month-to-month",
        "tenure_months": 3,
        "monthly_charges": 89.50,
    },
    {
        "customer_id": "C-1002",
        "name": "David Kim",
        "churn_probability": 0.61,
        "top_risk_factors": [
            {"feature": "Monthly charges", "value": 104.20, "impact": 0.19},
            {"feature": "No online security add-on", "value": True, "impact": 0.15},
            {"feature": "Payment method", "value": "Electronic check", "impact": 0.09},
        ],
        "ticket_text": "Asked billing why the price went up again this quarter.",
        "ticket_sentiment": "neutral",
        "contract_type": "One year",
        "tenure_months": 14,
        "monthly_charges": 104.20,
    },
    {
        "customer_id": "C-1003",
        "name": "Priya Nair",
        "churn_probability": 0.12,
        "top_risk_factors": [
            {"feature": "Tenure", "value": "38 months", "impact": -0.22},
            {"feature": "Contract type", "value": "Two year", "impact": -0.18},
        ],
        "ticket_text": "No recent tickets.",
        "ticket_sentiment": "positive",
        "contract_type": "Two year",
        "tenure_months": 38,
        "monthly_charges": 65.00,
    },
]
