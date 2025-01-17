import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Ethereum Anomaly Detection Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("🔍 Ethereum Transaction Anomaly Detection")

# Sidebar
st.sidebar.header("Dashboard Controls")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(datetime.now() - timedelta(days=7), datetime.now())
)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Volume")
    # Placeholder for transaction volume chart
    chart_data = pd.DataFrame({
        'date': pd.date_range(start=date_range[0], end=date_range[1]),
        'volume': [100, 120, 80, 200, 150, 90, 110]
    })
    fig = px.line(chart_data, x='date', y='volume')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Anomaly Distribution")
    # Placeholder for anomaly distribution
    anomaly_data = pd.DataFrame({
        'score': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        'count': [50, 30, 20, 10, 5, 3, 1]
    })
    fig = px.histogram(anomaly_data, x='score', y='count')
    st.plotly_chart(fig, use_container_width=True)

# Recent Anomalies
st.subheader("Recent Anomalies")
anomalies = pd.DataFrame({
    'timestamp': pd.date_range(start=date_range[0], end=date_range[1]),
    'transaction_hash': ['0x123...'] * 7,
    'anomaly_score': [0.95, 0.92, 0.89, 0.88, 0.87, 0.86, 0.85]
})
st.dataframe(anomalies)
