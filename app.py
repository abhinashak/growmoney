import streamlit as st

st.set_page_config(layout="wide", page_title="Market Intelligence Suite")

st.title("Welcome to the Market Intelligence Suite")
st.markdown("""
### Select a tool from the sidebar to begin.
            
Currently available:
- **Stock Comparison**: Analyze relative performance and moving averages for Indian and Global assets.
- *More tools coming soon...*
""")

# You can add a global market summary or news feed here later
st.info("👈 Use the sidebar to navigate between different analysis modules.")
