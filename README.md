# Customer Churn AI Agent

An end-to-end AI agent system that predicts customer churn risk, explains the underlying behavioral risk drivers using model-native SHAP values, and generates personalized retention action plans for Customer Success teams.

---

## Problem Statement

Customer Success teams manage hundreds of customer accounts and often only discover a customer is unhappy after they have already decided to cancel. This project builds an AI agent that analyzes customer usage, support interactions, feedback, and account activity to identify customers at risk of churn, explain the warning signals behind that risk, and recommend personalized retention actions before accounts are lost.

---

## Approach

The system uses a 3-stage pipeline: **Predict → Explain → Recommend**

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   1. PREDICT    │  ───> │   2. EXPLAIN    │  ───> │   3. RECOMMEND  │
│ XGBoost Model   │       │  SHAP Analytics │       │   LLM CS Agent  │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

1. **PREDICT:** A trained XGBoost classifier computes exact churn probabilities for each customer based on structured behavioral, usage, support, and account data.
2. **EXPLAIN:** SHAP (SHapley Additive exPlanations) decomposes each customer's prediction into the exact features driving their risk score, grounding every explanation in empirical model math rather than LLM guesswork.
3. **RECOMMEND:** An LLM agent (via Pollinations.ai's OpenAI-compatible API) ingests the SHAP-grounded risk drivers and generates a plain-English explanation for CS reps, along with 2–3 concrete, personalized retention steps tied to those exact risk factors.

---

## Dataset

We use the **"Customer Churn Prediction Business Dataset"** (Kaggle, by miadul) containing 10,000 synthetic B2B customer records across four key operational areas:

- **Usage:** `monthly_logins`, `weekly_active_days`, `avg_session_time`, `features_used`, `usage_growth_rate`, `last_login_days_ago`
- **Support:** `support_tickets`, `avg_resolution_time`, `complaint_type`, `csat_score`, `escalations`
- **Feedback:** `nps_score`, `survey_response`
- **Account Activity:** `tenure_months`, `contract_type`, `monthly_fee`, `total_revenue`, `payment_method`, `payment_failures`, `discount_applied`

> **Why Business Churn over Telco Churn?**  
> We deliberately selected this B2B dataset over the standard Telco dataset because it includes all four data categories required by our problem statement (usage telemetry, support metrics, feedback scores, and billing activity), whereas Telco is limited almost entirely to account and billing attributes.

---

## Model Performance (Final & Locked)

- **Algorithm:** XGBoost Classifier
- **Class Imbalance Strategy:** `scale_pos_weight = 8.79` (derived from majority-to-minority class ratio in training data)
- **Decision Threshold:** `0.44` (tuned via Precision-Recall curve analysis)
- **Evaluation Set:** Untouched held-out test set of $N=2,000$ customers

### Final Test Metrics

| Metric | Score | Impact |
| :--- | :---: | :--- |
| **Recall (Churn Class)** | **70.1%** | Catches 143 out of 204 actual churners |
| **Precision (Churn Class)** | **25.4%** | 1 in 4 flagged customers is a true churner |
| **F1-Score (Churn Class)** | **37.3%** | Harmonic mean at optimal decision threshold |
| **Overall Accuracy** | **76.0%** | Overall classification accuracy across all accounts |

### Design Rationale

In churn prevention, missing an at-risk customer results in a permanently lost account and lost recurring revenue, whereas a false positive only costs a CS representative a single proactive check-in call. We intentionally optimized decision thresholds to maximize **Recall (70.1%)**, prioritizing early detection over aggressive false-alarm suppression.

---

## What Makes This Different

1. **SHAP-Grounded Explanations:** Explanations are driven strictly by exact SHAP feature contributions rather than unconstrained LLM hallucination.
2. **Problem-Aligned Dataset Selection:** Chosen specifically to reflect real B2B SaaS telemetry (logins, support tickets, CSAT, NPS) rather than simple Telco billing records.
3. **Intentional Precision/Recall Trade-off:** Decision thresholds were tuned based on business unit economics—preferring high recall over arbitrary 0.5 probability cutoffs.

---

## Tech Stack

- **Core & Data:** Python, pandas, numpy
- **Machine Learning:** scikit-learn, XGBoost
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **LLM Agent & API:** Pollinations.ai API (OpenAI-compatible endpoint)
- **Frontend & App:** Streamlit, Custom HTML/CSS/JS dashboard UI

---

## Project Structure

```text
Churn/
├── app/
│   ├── dashboard.html          # Embedded HTML/CSS/JS interactive dashboard UI
│   └── streamlit_app.py        # Streamlit application host
├── data/
│   ├── raw/
│   │   └── business_churn.csv  # Raw Kaggle business churn dataset
│   └── processed/
│       └── dashboard_data.json # Pre-generated JSON profiles for dashboard
├── models/
│   └── churn_model.pkl         # Trained XGBoost model artifact (locked)
├── src/
│   ├── agent/
│   │   ├── risk_explainer.py   # LLM agent for SHAP risk explanation
│   │   └── retention_actions.py# LLM agent for factor-specific retention steps
│   ├── data/
│   │   └── preprocess.py       # Data cleaning & column transformation
│   ├── features/
│   │   └── build_features.py   # One-hot encoding & feature matrix construction
│   └── models/
│       ├── train.py            # XGBoost training pipeline with scale_pos_weight
│       ├── explain.py          # SHAP TreeExplainer calculation
│       └── predict.py          # Threshold-aware inference engine (threshold=0.44)
├── scripts/
│   ├── run_pipeline.py         # End-to-end customer risk profile execution
│   ├── generate_dashboard_data.py # Pre-computes dashboard JSON data
│   └── build_static_dashboard.py  # Generates standalone app/dashboard_final.html
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
└── README.md                   # Project documentation
```

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Pranjalhehe/customer-churn-ai-agent.git
   cd customer-churn-ai-agent
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Add your `POLLINATIONS_API_KEY` inside `.env` (obtain a key at [enter.pollinations.ai/keys](https://enter.pollinations.ai/keys)).

4. **Add the raw dataset:**
   Ensure `business_churn.csv` is placed in `data/raw/business_churn.csv`.

5. **Train the model, generate dashboard data, and build standalone HTML:**
   ```bash
   python src/models/train.py
   python scripts/generate_dashboard_data.py
   python scripts/build_static_dashboard.py
   ```

6. **Open the standalone dashboard:**
   Double-click `app/dashboard_final.html` to open directly in any browser (no server or Streamlit required).

---

## Project Status

Built for a hackathon in a 30-hour timeframe.
