import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Model Performance")

data = {
    "Model": ["LSTM", "XGBoost", "CNN", "Linear Regression"],
    "MAE": [3200, 3000, 3400, 4100],
    "RMSE": [4500, 4300, 4800, 5200]
}

df = pd.DataFrame(data)

fig = px.bar(df, x="Model", y="MAE", title="Model MAE Comparison")

st.plotly_chart(fig)

st.table(df)
