import os
import sys
import re
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

def build_actions_prompt(risk_explanation: str, top_risk_factors: Optional[list] = None) -> str:
    """Builds a factor-specific prompt for business retention actions."""
    factors_summary = ""
    if top_risk_factors:
        factors_summary = "\nTop Risk Drivers:\n" + "\n".join([
            f"- {f.get('feature', '')}: {f.get('value', '')} ({f.get('direction', '')})"
            for f in top_risk_factors
        ])

    return f"""You are an expert B2B Customer Retention Strategist.
Based on the following customer churn risk profile:
- Explanation: "{risk_explanation}"{factors_summary}

Suggest exactly 2-3 specific, highly actionable retention steps that a Customer Success Manager can execute THIS WEEK.

CRITICAL INSTRUCTIONS:
1. Base your recommendations SPECIFICALLY and ONLY on the risk drivers mentioned above (e.g. low monthly_logins -> user onboarding / training session; low csat_score -> senior CS support sprint; payment_failures -> billing audit; low marketing_click_rate -> product re-engagement).
2. Do NOT mention Telco concepts like "Fiber Optic", "month-to-month contract", or "phone lines".
3. Write concrete, tactical steps for a B2B SaaS / Business product context.

Format your response strictly as a numbered list (1., 2., 3.) with one action per item."""

def is_genuine_risk_factor(f: dict) -> bool:
    """Check if a factor is a genuine risk driver (not protective or 0-count)."""
    direction = f.get('direction', '')
    shap_val = f.get('shap_value', 0)
    feat = f.get('feature', '')
    val = f.get('value', 0)
    
    if direction == 'decreased risk' or shap_val < 0:
        return False
    if feat == 'payment_failures' and val < 1:
        return False
    if feat == 'csat_score' and val >= 5:
        return False
    if feat == 'nps_score' and val >= 6:
        return False
    if feat == 'monthly_logins' and val >= 10:
        return False
    if feat == 'escalations' and val < 1:
        return False
    return True

def recommend_actions(
    risk_explanation: str, 
    top_risk_factors: Optional[list] = None, 
    risk_level: str = 'Medium',
    customer_context: Optional[dict] = None, 
    api_key: Optional[str] = None
) -> List[str]:
    """
    Recommends 2-3 concrete, actionable retention steps tailored specifically to customer risk drivers.
    For Low-risk customers with no active risk drivers, returns light-touch expansion and partnership actions.
    """
    # Filter to genuine risk drivers
    genuine_risk_factors = [f for f in (top_risk_factors or []) if is_genuine_risk_factor(f)]
    
    # Handle Low-risk / happy customers
    if risk_level == 'Low' or (not genuine_risk_factors and 'low' in risk_explanation.lower()):
        return [
            "Maintain current engagement check-ins and share upcoming quarterly product roadmap updates.",
            "Evaluate account readiness for enterprise feature expansion or customer advocacy nomination.",
            "Schedule routine 90-day executive alignment check-in to reinforce partnership goals."
        ]
        
    key = api_key or os.getenv("POLLINATIONS_API_KEY")
    prompt = build_actions_prompt(risk_explanation, genuine_risk_factors)
    
    text = ""
    if not key or key.strip() == "" or key == "your_api_key_here":
        print("\n⚠️ [AGENT WARNING] POLLINATIONS_API_KEY is not set in .env! Using dynamic factor-based fallback for recommend_actions().", file=sys.stderr)
    else:
        try:
            client = OpenAI(
                api_key=key,
                base_url="https://gen.pollinations.ai/v1"
            )
            response = client.chat.completions.create(
                model="openai",
                messages=[{"role": "user", "content": prompt}]
            )
            raw_text = response.choices[0].message.content
            if raw_text and raw_text.strip():
                text = raw_text.strip()
        except Exception as e:
            print(f"\n❌ [AGENT API ERROR] Pollinations API call failed in recommend_actions(): {e}! Using dynamic fallback actions.", file=sys.stderr)

    if text:
        actions = []
        for line in text.splitlines():
            cleaned = re.sub(r'^\s*(\d+[\.\)]|[\-\*\•])\s*', '', line).strip()
            if cleaned:
                actions.append(cleaned)
        if actions:
            return actions[:3]

    # Dynamic local fallback tied strictly to genuine risk factors
    fallback_actions = []
    for f in genuine_risk_factors:
        feat = f.get('feature', '')
        val = f.get('value', 0)
        if (feat == 'monthly_logins' or feat == 'weekly_active_days') and val < 10:
            fallback_actions.append(f"Schedule a 1-on-1 Product Onboarding & Activation session to help the user rebuild regular login habits (currently {int(val)} logins/month).")
        elif (feat == 'csat_score' or feat == 'survey_response_Unsatisfied') and val < 5:
            fallback_actions.append(f"Assign a Senior CS lead for a 48-hour Sentiment Recovery Sprint to address root causes behind the low CSAT score ({val}/10).")
        elif feat == 'payment_failures' and val >= 1:
            fallback_actions.append(f"Reach out to update payment credentials and offer flexible billing terms to resolve recent payment failures ({int(val)} failed attempts).")
        elif (feat == 'support_tickets' or feat == 'escalations' or feat == 'complaint_type_Technical') and val >= 1:
            fallback_actions.append("Schedule a joint Technical Account Manager call to review and resolve open support tickets and technical escalations.")
        elif (feat == 'tenure_months' or feat == 'tenure') and val < 12:
            fallback_actions.append("Provide a complimentary 90-day feature expansion trial and dedicated CSM check-in to strengthen early account stickiness.")
        elif feat == 'monthly_fee' or feat == 'price_increase_last_3m':
            fallback_actions.append(f"Offer a custom pricing audit or temporary 10-15% loyalty credit to alleviate monthly bill pressure (${val:.2f}).")
            
    if len(fallback_actions) < 2:
        if risk_level == 'High':
            fallback_actions.append("Schedule an urgent Executive Sponsor check-in call to review account health and establish a joint success plan.")
            fallback_actions.append("Provide a 30-day priority support SLA and product re-activation training for key user teams.")
        else:
            fallback_actions.append("Schedule a CSM quarterly account review to optimize workspace usage and review key workflows.")
            fallback_actions.append("Share targeted training resources and feature documentation with team leads.")

    return fallback_actions[:3]
