import os
import json
import streamlit as st

st.set_page_config(layout="wide")

# Modern Streamlit CSS targeting exact data-testid attributes to hide toolbar/header/footer
custom_css = """
    <style>
        [data-testid="stToolbar"] {visibility: hidden !important;}
        [data-testid="stHeader"] {visibility: hidden !important;}
        [data-testid="stDecoration"] {display: none !important;}
        #MainMenu {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        
        .stApp {
            background-color: #F3F5FA !important;
        }
        .main .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
            max-width: 100% !important;
        }
        div[data-testid="stVerticalBlock"] {
            gap: 0rem !important;
        }
        iframe {
            width: 100% !important;
            border: none !important;
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def load_data():
    candidates = [
        os.path.abspath('data/processed/dashboard_data.json'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed', 'dashboard_data.json'),
        os.path.abspath('../data/processed/dashboard_data.json')
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None

def main():
    customer_data = load_data()
    if not customer_data:
        st.error("Data file not found in data/processed/dashboard_data.json. Please run 'python scripts/generate_dashboard_data.py' first.")
        return
        
    v2_html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_v2.html')
    v1_html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard.html')
    
    html_path = v2_html_path if os.path.exists(v2_html_path) else v1_html_path
    
    with open(html_path, 'r', encoding='utf-8') as f:
        template_html = f.read()
        
    json_str = json.dumps(customer_data)
    final_html = template_html.replace('__CUSTOMER_DATA__', json_str)
    
    st.components.v1.html(final_html, height=1200, scrolling=True)

if __name__ == '__main__':
    main()
