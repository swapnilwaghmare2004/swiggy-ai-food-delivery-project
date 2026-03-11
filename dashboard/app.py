import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Swiggy AI Dashboard", layout="wide")

st.title("🍔 Swiggy AI Food Delivery Analytics Dashboard")

# Load dataset
df = pd.read_csv("data/orders.csv")

# Sidebar
st.sidebar.header("Filters")

hour_filter = st.sidebar.slider(
    "Select Order Hour",
    int(df.order_hour.min()),
    int(df.order_hour.max()),
    (0,23)
)

filtered_df = df[(df.order_hour >= hour_filter[0]) & (df.order_hour <= hour_filter[1])]

# Metrics
col1,col2,col3 = st.columns(3)

col1.metric("Total Orders", len(filtered_df))
col2.metric("Average Delivery Time", round(filtered_df.delivery_time.mean(),2))
col3.metric("Average Order Value", round(filtered_df.order_value.mean(),2))

st.divider()

# Orders by hour
st.subheader("Orders by Hour")

orders_hour = df.groupby("order_hour").size()

st.bar_chart(orders_hour)

# Delivery time vs distance
st.subheader("Delivery Time vs Distance")

st.scatter_chart(
    df,
    x="distance_km",
    y="delivery_time"
)

# Order value distribution
st.subheader("Order Value Distribution")

st.line_chart(df["order_value"])

# Raw data
with st.expander("View Raw Dataset"):
    st.dataframe(df)

st.success("Dashboard Loaded Successfully")