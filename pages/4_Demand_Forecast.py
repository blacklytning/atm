import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import load_data, add_rolling_features, encode_categorical
from utils.predict import forecast_next_days

st.title("📈 ATM Demand Forecast")

df = load_data("data/atm_data.csv")

df = add_rolling_features(df)
df, encoders = encode_categorical(df)

atm = st.selectbox(
    "Select ATM",
    df["atm_id"].unique()
)

atm_df = df[df["atm_id"] == atm]

if st.button("Generate 7-Day Forecast"):

    forecast = forecast_next_days(atm_df)

    forecast_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(7)],
        "Predicted Demand": forecast
    })

    st.subheader("Next 7 Days Forecast")

    st.table(forecast_df)

    fig = px.line(
        forecast_df,
        x="Day",
        y="Predicted Demand",
        markers=True,
        title="7 Day ATM Cash Demand Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)
