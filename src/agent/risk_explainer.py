import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def translate_feature_name(feature: str, value: float, direction: str) -> str:
    """Translates raw model feature names and values into plain business language."""
    if feature == "tenure_months" or feature == "tenure":
        return f"Customer tenure is short ({int(value)} month(s))" if value < 12 else f"Customer tenure is {int(value)} months"
    elif feature == "csat_score":
        return f"CSAT satisfaction score is low ({int(value)}/10)" if value < 6 else f"CSAT score is {int(value)}/10"
    elif feature == "payment_failures":
        return f"Experienced {int(value)} payment failure(s)"
    elif feature == "monthly_logins":
        return f"Monthly logins are low ({int(value)} logins/month)" if value < 10 else f"Monthly logins count is {int(value)}"
    elif feature == "last_login_days_ago":
        return f"Last active {int(value)} day(s) ago"
    elif feature == "monthly_fee" or feature == "MonthlyCharges":
        return f"Monthly fee is ${value:.2f}"
    elif feature == "total_revenue" or feature == "TotalCharges":
        return f"Total lifetime revenue is ${value:.2f}"
    elif feature == "contract_type_Yearly":
        return "Customer is on a Yearly contract" if value == 1 else "Customer is on a Month-to-Month plan (no annual contract)"
    elif feature == "contract_type_Quarterly":
        return "Customer is on a Quarterly contract" if value == 1 else "Customer is not on a Quarterly contract"
    elif feature == "complaint_type_Technical":
        return "Filed a technical complaint" if value == 1 else "No technical complaint filed"
    elif feature == "complaint_type_Service":
        return "Filed a service quality complaint" if value == 1 else "No service complaint filed"
    elif feature == "complaint_type_None":
        return "No recent customer support complaints" if value == 1 else "Has customer complaints on file"
    elif feature == "survey_response_Unsatisfied":
        return "Survey response was Unsatisfied" if value == 1 else "Survey response was not Unsatisfied"
    elif feature == "survey_response_Satisfied":
        return "Survey response was Satisfied" if value == 1 else "Survey response was not Satisfied"
    elif feature == "discount_applied":
        return "Has active discount" if value == 1 else "No active discount applied"
    elif feature == "price_increase_last_3m":
        return "Experienced price increase in last 3 months" if value == 1 else "No recent price increase"
    else:
        clean_name = feature.replace('_', ' ')
        return f"{clean_name} = {value}"

def build_explanation_prompt(shap_output: dict) -> str:
    """Constructs a strict, factor-specific prompt for the LLM."""
    churn_prob = shap_output.get("churn_probability", 0.0)
    top_factors = shap_output.get("top_risk_factors", [])
    
    translated_factors = [
        f"#{i+1}: {translate_feature_name(f['feature'], f['value'], f['direction'])} (SHAP Impact: {f['shap_value']:+.4f}, Direction: {f['direction']})"
        for i, f in enumerate(top_factors)
    ]
    factors_str = "\n".join(translated_factors)
    
    primary_factor = translate_feature_name(top_factors[0]['feature'], top_factors[0]['value'], top_factors[0]['direction']) if top_factors else "account status"
    
    prompt = f"""You are an AI assistant for a Customer Success Representative.
A customer churn risk model evaluated a customer and found:
- Predicted Churn Probability: {churn_prob:.1%}
- Top Key Risk Factors (ordered strictly by importance/impact):
{factors_str}

Write a short, 2-3 sentence, plain-English explanation of why this customer is at risk of churning.

CRITICAL INSTRUCTIONS:
1. Base your explanation SPECIFICALLY and PRIMARILY on the #1 top factor: "{primary_factor}".
2. Explicitly mention the specific numbers given (e.g. specific monthly fee, CSAT score, or tenure).
3. Do NOT use generic churn template language. Customize the emphasis completely based on the #1 risk factor above.
4. Do NOT use data science jargon like "SHAP", "features", "variables", or "model weights".
5. Write strictly for a Customer Success Representative."""

    return prompt

def explain_risk(shap_output: dict, api_key: str = None) -> str:
    """
    Generate a 2-3 sentence, plain-English explanation of customer churn risk
    tailored for a Customer Success Representative using Pollinations.ai API via OpenAI SDK.
    """
    key = api_key or os.getenv("POLLINATIONS_API_KEY")
    churn_prob = shap_output.get("churn_probability", 0.0)
    top_factors = shap_output.get("top_risk_factors", [])
    
    prompt = build_explanation_prompt(shap_output)
    
    if not key or key.strip() == "" or key == "your_api_key_here":
        print("\n⚠️ [AGENT WARNING] POLLINATIONS_API_KEY is not set in .env! Falling back to dynamic factor-based explanation.", file=sys.stderr)
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
            text = response.choices[0].message.content
            if text and text.strip():
                return text.strip()
        except Exception as e:
            print(f"\n❌ [AGENT API ERROR] Pollinations API call failed: {e}! Falling back to dynamic factor-based explanation.", file=sys.stderr)

    # Dynamic fallback based on customer's exact top SHAP factors
    if top_factors:
        f1_desc = translate_feature_name(top_factors[0]['feature'], top_factors[0]['value'], top_factors[0]['direction'])
        f2_desc = translate_feature_name(top_factors[1]['feature'], top_factors[1]['value'], top_factors[1]['direction']) if len(top_factors) > 1 else ""
        return (
            f"This customer faces a high churn probability of {churn_prob:.1%}, driven primarily because {f1_desc.lower()}. "
            f"{'Additionally, ' + f2_desc.lower() + '.' if f2_desc else ''} "
            f"Immediate outreach is recommended to mitigate risk."
        )
    
    return f"This customer is at {churn_prob:.1%} risk of churning. Proactive account review is recommended."
