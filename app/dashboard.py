"""
Customer Churn Risk Agent — Streamlit dashboard.

Run with:
    streamlit run app/dashboard.py

Requires ANTHROPIC_API_KEY set in your environment.
"""

import streamlit as st
import pandas as pd

from mock_data import MOCK_CUSTOMERS
from prompts import explain_risk, recommend_actions

st.set_page_config(page_title="Churn Risk Agent", layout="wide")

st.title("Customer Churn Risk Agent")
st.caption("At-risk customers, ranked by predicted churn probability.")

# --- Table of customers, sorted highest risk first ---
customers_sorted = sorted(MOCK_CUSTOMERS, key=lambda c: c["churn_probability"], reverse=True)

table_df = pd.DataFrame([
    {
        "Customer": c["name"],
        "ID": c["customer_id"],
        "Churn Risk": f"{c['churn_probability']:.0%}",
        "Contract": c["contract_type"],
        "Tenure (mo)": c["tenure_months"],
    }
    for c in customers_sorted
])

st.dataframe(table_df, use_container_width=True, hide_index=True)

st.divider()

# --- Detail panel: pick a customer, see the AI explanation + recommendations ---
st.subheader("Customer Detail")

selected_name = st.selectbox(
    "Select a customer to analyze",
    options=[c["name"] for c in customers_sorted],
)
selected_customer = next(c for c in customers_sorted if c["name"] == selected_name)

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Churn Probability", f"{selected_customer['churn_probability']:.0%}")
    st.write("**Top Risk Factors**")
    for f in selected_customer["top_risk_factors"]:
        arrow = "🔺" if f["impact"] > 0 else "🔻"
        st.write(f"{arrow} {f['feature']}: {f['value']}")
    st.write("**Recent Ticket**")
    st.write(f"_{selected_customer['ticket_text']}_")

with col2:
    if st.button("Generate AI Analysis", type="primary"):
        with st.spinner("Analyzing risk..."):
            explanation = explain_risk(selected_customer)
        st.write("**Why they're at risk**")
        st.write(explanation)

        with st.spinner("Generating retention plan..."):
            actions = recommend_actions(selected_customer, explanation)
        st.write("**Recommended actions**")
        st.write(actions)
    else:
        st.info("Click 'Generate AI Analysis' to run the risk explainer and retention agent.")
