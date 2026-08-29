"""
The two Claude-powered agent functions:
1. explain_risk()      -> plain-English reason a customer is at risk
2. recommend_actions()  -> 2-3 concrete retention steps

Both call the Anthropic API directly. Requires ANTHROPIC_API_KEY as an
environment variable (never hardcode the key in the file).
"""

import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-5"  # swap freely; sonnet is a good cost/quality tradeoff for a hackathon


def _format_risk_factors(top_risk_factors: list[dict]) -> str:
    """Turn the SHAP-style list into readable bullet text for the prompt."""
    lines = []
    for f in top_risk_factors:
        direction = "increases" if f["impact"] > 0 else "decreases"
        lines.append(f"- {f['feature']}: {f['value']} ({direction} risk, impact {abs(f['impact']):.2f})")
    return "\n".join(lines)


def explain_risk(customer: dict) -> str:
    """
    Given one customer dict (see mock_data.py for shape), return a 2-3 sentence
    plain-English explanation of why they're at risk.
    """
    prompt = f"""You are a customer success analyst. Given a customer's churn risk data,
explain in 2-3 plain-English sentences why they're at risk. Be specific
and reference the actual data points, not generic statements.

Customer data:
- Churn probability: {customer['churn_probability']:.0%}
- Top risk factors (from model):
{_format_risk_factors(customer['top_risk_factors'])}
- Recent support ticket: {customer['ticket_text']}
- Ticket sentiment: {customer['ticket_sentiment']}

Write the explanation as if briefing a CS rep who has 10 seconds to read it
before a call. No fluff, no restating the probability number, just the "why."
"""
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def recommend_actions(customer: dict, risk_explanation: str) -> str:
    """
    Given the risk explanation from explain_risk(), return a short numbered
    list of 2-3 concrete retention actions.
    """
    prompt = f"""You are a customer success strategist. Based on this risk explanation,
recommend exactly 2-3 specific, actionable retention steps a CS rep could
take this week. Avoid generic advice like "reach out to the customer" —
tie each recommendation directly to the risk factors mentioned.

Risk explanation:
{risk_explanation}

Customer context:
- Contract type: {customer['contract_type']}
- Tenure: {customer['tenure_months']} months
- Monthly charges: ${customer['monthly_charges']:.2f}

Format as a short numbered list. Each item: one action + one sentence on why.
"""
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
