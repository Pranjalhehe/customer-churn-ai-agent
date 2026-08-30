import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def translate_feature_name(feature: str, value: float, direction: str) -> str:
    """Translates raw model feature names and values into plain business language."""
    if feature == "tenure_months" or feature == "tenure":
        if value < 12:
            return f"Customer tenure is short ({int(value)} mos)"
        elif value < 24:
            return f"Customer tenure is moderate ({int(value)} mos)"
        else:
            return f"Customer tenure is long ({int(value)} mos)"
    elif feature == "csat_score":
        if value <= 4:
            return f"CSAT satisfaction score is low ({int(value)}/10)"
        elif value <= 7:
            return f"CSAT satisfaction score is average ({int(value)}/10)"
        else:
            return f"CSAT satisfaction score is high ({int(value)}/10)"
    elif feature == "nps_score":
        if value <= 5:
            return f"NPS score is low ({int(value)}/10)"
        elif value <= 8:
            return f"NPS score is average ({int(value)}/10)"
        else:
            return f"NPS score is high ({int(value)}/10)"
    elif feature == "payment_failures":
        return f"Experienced {int(value)} payment failure(s)"
    elif feature == "monthly_logins":
        if value < 5:
            return f"Monthly logins are low ({int(value)} logins/month)"
        elif value <= 15:
            return f"Monthly logins are moderate ({int(value)} logins/month)"
        else:
            return f"Monthly logins are high ({int(value)} logins/month)"
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
    ticket_excerpt = shap_output.get("support_ticket_excerpt", "")
    feedback_snippet = shap_output.get("feedback_snippet", "")
    
    risk_level = shap_output.get("risk_level")
    if not risk_level:
        if churn_prob > 0.44:
            risk_level = "High"
        elif churn_prob >= 0.20:
            risk_level = "Medium"
        else:
            risk_level = "Low"
    
    translated_factors = [
        f"#{i+1}: {translate_feature_name(f['feature'], f['value'], f['direction'])} (SHAP Impact: {f['shap_value']:+.4f}, Direction: {f['direction']})"
        for i, f in enumerate(top_factors)
    ]
    factors_str = "\n".join(translated_factors)
    
    primary_factor = translate_feature_name(top_factors[0]['feature'], top_factors[0]['value'], top_factors[0]['direction']) if top_factors else "account status"
    
    prompt = f"""You are an AI assistant for a Customer Success Representative.
A customer churn risk model evaluated a customer and found:
- Assigned Risk Level: {risk_level} (Predicted Churn Probability: {churn_prob:.1%})
- Top Key Risk Factors (ordered strictly by importance/impact):
{factors_str}
- Support Ticket Excerpt: "{ticket_excerpt}"
- Customer Feedback Snippet: "{feedback_snippet}"

Write a short, 2-3 sentence, plain-English explanation of why this customer is at risk of churning.

CRITICAL INSTRUCTIONS:
1. Base your explanation SPECIFICALLY and PRIMARILY on the #1 top factor: "{primary_factor}".
2. Reference the customer's specific support ticket excerpt or feedback snippet directly (e.g., "Their recent ticket reported unresolved billing issues..." or "Customer survey feedback noted...").
3. Do NOT use generic churn template language. Customize the emphasis completely based on the #1 risk factor and customer text above.
4. Do NOT use data science jargon like "SHAP", "features", "variables", or "model weights".
5. Tailor the closing sentence strictly to the customer's assigned Risk Level ({risk_level}):
   - If Risk Level is High: include an urgent action recommendation (e.g. "Immediate CS outreach is recommended to mitigate churn risk.").
   - If Risk Level is Medium: suggest a proactive CSM check-in (e.g. "A proactive CSM check-in is recommended to discuss performance and address minor concerns."). Do NOT use urgent phrases like "Immediate CS outreach is recommended".
   - If Risk Level is Low: do NOT use urgent language or "immediate outreach" calls to action. Instead, state that the account is in healthy standing with routine monitoring.
6. Write strictly for a Customer Success Representative."""

    return prompt

HIGH_RISK_CLOSING_PHRASES = [
    "Immediate CS outreach is recommended to mitigate churn risk.",
    "Urgent Customer Success intervention is required to stabilize account health.",
    "Priority executive alignment and dedicated outreach are strongly advised.",
    "Prompt CSM engagement should be prioritized before contract renewal."
]

def get_high_risk_closing(shap_output: dict) -> str:
    cid = str(shap_output.get("customer_id", ""))
    top_factors = shap_output.get("top_risk_factors", [])
    primary_feat = top_factors[0]["feature"] if top_factors else ""
    seed_str = cid + primary_feat
    idx = abs(hash(seed_str)) % len(HIGH_RISK_CLOSING_PHRASES)
    return HIGH_RISK_CLOSING_PHRASES[idx]

import time

GROQ_MODELS_POOL = ["groq/compound-mini", "qwen/qwen3.6-27b", "openai/gpt-oss-20b", "groq/compound"]

def call_groq_with_retry(client, model_name, prompt, max_retries=2):
    models_to_try = [model_name] + [m for m in GROQ_MODELS_POOL if m != model_name]
    last_err = None
    for target_model in models_to_try:
        for attempt in range(max_retries):
            try:
                time.sleep(0.2)
                print(f"   -> [LLM Request] Model='{target_model}' (Attempt {attempt+1}/{max_retries})...", flush=True)
                response = client.chat.completions.create(
                    model=target_model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=12.0
                )
                print(f"   [OK] [LLM Response Success]", flush=True)
                return response
            except Exception as e:
                last_err = e
                err_msg = str(e).lower()
                print(f"   [WARN] [LLM Attempt Failed] Model='{target_model}' Error: {e}", flush=True)
                if "rate_limit" in err_msg or "429" in err_msg or "rpd" in err_msg or "tpd" in err_msg or "404" in err_msg:
                    print(f"   -> Rate limit/quota reached on '{target_model}'. Switching to next model in pool...", flush=True)
                    break
                time.sleep(1.0 * (attempt + 1))
    if last_err:
        raise last_err

def explain_risk(shap_output: dict, api_key: str = None) -> str:
    """
    Generate a 2-3 sentence, plain-English explanation of customer churn risk
    tailored for a Customer Success Representative using Groq API via OpenAI SDK.
    """
    key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("POLLINATIONS_API_KEY")
    churn_prob = shap_output.get("churn_probability", 0.0)
    top_factors = shap_output.get("top_risk_factors", [])
    ticket_excerpt = shap_output.get("support_ticket_excerpt", "")
    feedback_snippet = shap_output.get("feedback_snippet", "")
    
    prompt = build_explanation_prompt(shap_output)
    
    if not key or key.strip() == "" or key == "your_api_key_here":
        print("\n[WARNING] GROQ_API_KEY is not set in .env! Falling back to dynamic factor-based explanation.", file=sys.stderr)
    else:
        try:
            client = OpenAI(
                api_key=key,
                base_url="https://api.groq.com/openai/v1"
            )
            model_name = os.getenv("GROQ_MODEL", "groq/compound-mini")
            response = call_groq_with_retry(client, model_name, prompt)
            text = response.choices[0].message.content
            if text and text.strip():
                return text.strip()
        except Exception as e:
            print(f"\n[AGENT API ERROR] Groq API call failed: {e}! Falling back to dynamic factor-based explanation.", file=sys.stderr)

    # Dynamic fallback tailored strictly by risk level
    if top_factors:
        f1_desc = translate_feature_name(top_factors[0]['feature'], top_factors[0]['value'], top_factors[0]['direction'])
        
        if churn_prob >= 0.44:
            explanation = f"This customer faces a high churn risk of {churn_prob:.1%}, driven primarily because {f1_desc.lower()}."
            if ticket_excerpt:
                explanation += f" Their latest ticket notes: '{ticket_excerpt}'"
            elif feedback_snippet:
                explanation += f" Recent feedback highlighted: '{feedback_snippet}'"
            closing = get_high_risk_closing(shap_output)
            explanation += f" {closing}"
        elif churn_prob >= 0.20:
            explanation = f"This customer shows moderate churn risk at {churn_prob:.1%}, influenced by {f1_desc.lower()}."
            if ticket_excerpt:
                explanation += f" Recent ticket activity notes: '{ticket_excerpt}'"
            elif feedback_snippet:
                explanation += f" Customer feedback mentioned: '{feedback_snippet}'"
            explanation += " Proactive CSM check-in is recommended prior to renewal."
        else:
            explanation = f"This customer maintains a low churn risk of {churn_prob:.1%}. Account activity remains healthy"
            if feedback_snippet:
                explanation += f", with recent feedback noting: '{feedback_snippet}'"
            elif ticket_excerpt:
                explanation += f", and ticket logs showing: '{ticket_excerpt}'"
            explanation += "."
            
        return explanation
    
    if churn_prob >= 0.44:
        closing = get_high_risk_closing(shap_output)
        return f"This customer is at high risk ({churn_prob:.1%}). {closing}"
    elif churn_prob >= 0.20:
        return f"This customer is at moderate risk ({churn_prob:.1%}). Proactive CSM check-in recommended."
    else:
        return f"This customer maintains a low churn risk of {churn_prob:.1%} with stable account health."
