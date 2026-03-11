import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🗺 ATM Demand Map")

df = pd.read_csv("data/atm_data.csv")

# Dummy coordinates for visualization
import numpy as np
df["lat"] = 19.0 + np.random.rand(len(df)) * 0.1
df["lon"] = 72.8 + np.random.rand(len(df)) * 0.1

fig = px.scatter_mapbox(
    df,
    lat="lat",
    lon="lon",
    color="Cash_Demand_Next_Day",
    size="Cash_Demand_Next_Day",
    hover_name="ATM_ID",
    zoom=10,
)

fig.update_layout(mapbox_style="open-street-map")

st.plotly_chart(fig, use_container_width=True)
