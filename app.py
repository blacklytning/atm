import streamlit as st

st.set_page_config(
    page_title="ATM Cash Demand Forecasting",
    layout="wide"
)

st.title("🏧 ATM Cash Demand Forecasting Dashboard")

st.markdown("""
### Dashboard Pages

Use the sidebar to explore:

• Overview  
• ATM Map  
• ATM Details  
• Model Performance  
• Demand Forecast  

This system predicts **next-day ATM cash demand** using:

- Linear Regression
- Random Forest
- XGBoost
- LSTM
- CNN

Final predictions use a **Voting Ensemble**.
""")