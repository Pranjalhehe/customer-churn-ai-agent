import os
import sys
import io
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.upload_demo import process_uploaded_csv, validate_csv_schema
from src.models.explain import ChurnExplainer

class MockUploadedFile:
    def __init__(self, filepath):
        self.name = os.path.basename(filepath)
        with open(filepath, 'rb') as f:
            self._bytes = f.read()
        self.size = len(self._bytes)

    def getvalue(self):
        return self._bytes

def test_upload_demo():
    print("================ TESTING APP/UPLOAD_DEMO.PY END-TO-END ================")
    csv_path = 'demo_samples/fresh_customers.csv'
    assert os.path.exists(csv_path), f"File {csv_path} not found!"
    
    mock_file = MockUploadedFile(csv_path)
    
    # 1. Test schema validation with invalid columns
    print("\n--- 1. Testing Schema Validation with Invalid CSV ---")
    bad_df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    explainer = ChurnExplainer('models/churn_model.pkl')
    is_valid, missing_cols = validate_csv_schema(bad_df, list(explainer.model.feature_names_in_))
    print(f"Is valid: {is_valid}")
    print(f"Missing columns count: {len(missing_cols)}")
    assert not is_valid, "Expected schema validation to fail for invalid CSV!"
    assert len(missing_cols) > 0
    print("✅ Schema validation correctly caught missing columns without crashing.")
    
    # 2. Test full end-to-end processing with fresh_customers.csv
    print(f"\n--- 2. Processing {csv_path} through full pipeline ---")
    results, summary_stats = process_uploaded_csv(mock_file)
    
    assert results is not None, "Results should not be None!"
    assert summary_stats is not None, "Summary stats should not be None!"
    
    print("\n================ FULL RUN EXECUTIVE SUMMARY ================")
    print(f"Total Customers Processed : {summary_stats['total_processed']}")
    print(f"High Risk (🚨)           : {summary_stats['high_count']}")
    print(f"Medium Risk (⚠️)          : {summary_stats['med_count']}")
    print(f"Low Risk (✅)             : {summary_stats['low_count']}")
    print(f"Most Common Risk Driver   : {summary_stats['most_common_factor']}")
    print(f"Synthesis Phrase          : {summary_stats['summary_phrase']}")
    print("============================================================\n")
    
    print("================ RESULTS OVERVIEW TABLE (10 ROWS) ================")
    print(f"{'Customer ID':<15} | {'Risk %':<8} | {'Risk Level':<10} | {'Primary Risk Driver'}")
    print("-" * 65)
    for r in results:
        print(f"{r['customer_id']:<15} | {r['churn_probability']:<7.1%} | {r['risk_level']:<10} | {r['primary_factor']}")
    print("==================================================================\n")

    print("================ DETAIL SAMPLE (FIRST CUSTOMER EXPANDER) ================")
    c1 = results[0]
    print(f"Customer ID          : {c1['customer_id']}")
    print(f"Churn Probability    : {c1['churn_probability']:.1%}")
    print(f"Risk Tier            : {c1['risk_level']}")
    print("Top Risk Factors     :")
    for f in c1['top_risk_factors']:
        print(f"  - {f['feature']}: {f['value']} (SHAP: {f['shap_value']:+.4f}, {f['direction']})")
    print(f"Support Ticket       : {c1['support_ticket_excerpt']}")
    print(f"Customer Feedback    : {c1['feedback_snippet']}")
    print(f"AI Risk Explanation  : {c1['explanation']}")
    print("Recommended Actions  :")
    for idx, act in enumerate(c1['recommended_actions'], 1):
        print(f"  {idx}. {act}")
    print("=========================================================================\n")
    
    assert summary_stats['total_processed'] == 10, f"Expected 10 processed customers, got {summary_stats['total_processed']}"
    print("✅ All 10 customers successfully processed end-to-end!")

if __name__ == '__main__':
    test_upload_demo()
