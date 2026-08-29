import os
import sys
import json

def build_static_dashboard(
    json_path: str = 'data/processed/dashboard_data.json',
    template_path: str = 'app/dashboard_v2.html',
    output_path: str = 'app/dashboard_final.html'
):
    """
    Reads dashboard JSON data and injects it into app/dashboard_v2.html,
    saving the resulting standalone HTML file to app/dashboard_final.html.
    """
    # 1. Read JSON data
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Processed data file not found at {json_path}. Run generate_dashboard_data.py first.")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 2. Read template HTML
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template HTML file not found at {template_path}.")
        
    with open(template_path, 'r', encoding='utf-8') as f:
        template_html = f.read()
        
    # 3. Replace placeholder with JSON string
    json_str = json.dumps(data, indent=2)
    final_html = template_html.replace('__CUSTOMER_DATA__', json_str)
    
    # 4. Save to app/dashboard_final.html
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
        
    print(f"Successfully generated standalone HTML dashboard at {output_path} ({len(final_html)} bytes)")
    print(f"Embedded {len(data)} real customer risk profiles.")
    return output_path

if __name__ == '__main__':
    build_static_dashboard()
