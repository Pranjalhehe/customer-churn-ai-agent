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

def recommend_actions(risk_explanation: str, top_risk_factors: Optional[list] = None, customer_context: Optional[dict] = None, api_key: Optional[str] = None) -> List[str]:
    """
    Recommends 2-3 concrete, actionable retention steps tailored specifically to customer risk drivers.
    """
    key = api_key or os.getenv("POLLINATIONS_API_KEY")
    prompt = build_actions_prompt(risk_explanation, top_risk_factors)
    
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
            return actions

    # Dynamic local fallback tied strictly to the customer's top SHAP factors
    fallback_actions = []
    if top_risk_factors:
        for f in top_risk_factors:
            feat = f.get('feature', '')
            val = f.get('value', 0)
            if feat == 'monthly_logins' or feat == 'weekly_active_days':
                fallback_actions.append(f"Schedule a 1-on-1 Product Onboarding & Activation session to help the user rebuild regular login habits (currently {int(val)} logins/month).")
            elif feat == 'csat_score' or feat == 'survey_response_Unsatisfied':
                fallback_actions.append(f"Assign a Senior CS lead for a 48-hour Sentiment Recovery Sprint to address root causes behind the low CSAT score ({val}/10).")
            elif feat == 'payment_failures':
                fallback_actions.append(f"Reach out to update payment credentials and offer flexible billing terms to resolve recent payment failures ({int(val)} failed attempts).")
            elif feat == 'support_tickets' or feat == 'escalations' or feat == 'complaint_type_Technical':
                fallback_actions.append("Schedule a joint Technical Account Manager call to review and resolve open support tickets and technical escalations.")
            elif feat == 'tenure_months' or feat == 'tenure':
                fallback_actions.append("Provide a complimentary 90-day feature expansion trial and dedicated CSM check-in to strengthen early account stickiness.")
            elif feat == 'monthly_fee' or feat == 'price_increase_last_3m':
                fallback_actions.append(f"Offer a custom pricing audit or temporary 10-15% loyalty credit to alleviate monthly bill pressure (${val:.2f}).")
                
    if len(fallback_actions) < 2:
        fallback_actions.append("Schedule an urgent Executive Sponsor check-in call to review account health and establish a joint success plan.")
        fallback_actions.append("Provide a 30-day priority support SLA and product re-activation training for key user teams.")
        
    return fallback_actions[:3]
