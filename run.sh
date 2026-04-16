#!/bin/bash
# One-click runner for the Streamlit PCA app (non-interactive)
set -e
pip install -r requirements.txt --quiet
# disable onboarding prompt and run headless
export STREAMLIT_DISABLE_STREAMLIT_ONBOARDING=1
streamlit run streamlit_app.py --server.headless=true --server.address=0.0.0.0 --server.port=8501
