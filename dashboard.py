"""Three local Monte Carlo research modes; launch with python -m streamlit run dashboard.py."""

import streamlit as st

from src.legacy_dashboard import render_legacy
from src.research_dashboard import render_research

st.set_page_config(page_title="Portfolio Monte Carlo Lab", page_icon="📈", layout="wide",
                   initial_sidebar_state="auto")
st.markdown("""
<style>
  .block-container { max-width: 1480px; padding-top: 1.8rem; padding-bottom: 4rem; }
  [data-testid="stSidebar"] { border-right: 1px solid #263851; }
  .lab-eyebrow { color: #59d9cf; font-size: .77rem; letter-spacing: .18em; font-weight: 750; }
  .lab-title { font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1.12; font-weight: 780; margin: .45rem 0; }
  .lab-subtitle { color: #a7b8c9; max-width: 780px; line-height: 1.55; }
  .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 185px), 1fr)); gap: .75rem; margin: 1.2rem 0; }
  .metric-card { background: linear-gradient(145deg, #182940, #111e31); border: 1px solid #2c3e57; border-radius: 14px; padding: 1rem 1.1rem; min-width: 0; }
  .metric-label { color: #9eb3c7; font-size: .8rem; margin-bottom: .4rem; }
  .metric-value { color: #edf8fd; font-size: clamp(1.2rem, 2vw, 1.7rem); line-height: 1.2; font-weight: 700; overflow-wrap: anywhere; }
  .metric-note { color: #83a2b8; font-size: .72rem; margin-top: .28rem; }
  [data-testid="stAppDeployButton"] { display: none !important; }
  @media (max-width: 640px) {
    .block-container { padding: 1rem 1rem 3rem !important; }
  }
</style>
""", unsafe_allow_html=True)



mode = st.selectbox("Research mode", (
    "BTC / S&P 500 / Gold (original)", "Memecoin trading", "Spot portfolio",
), key="research_mode")
if mode == "BTC / S&P 500 / Gold (original)":
    render_legacy()
elif mode == "Memecoin trading":
    render_research("memecoin")
else:
    render_research("spot")
