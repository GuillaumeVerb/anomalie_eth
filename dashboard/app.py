import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

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
    value=(datetime.now(), datetime.now() + timedelta(days=7))
)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Volume")
    # Create sample data with proper date range
    dates = pd.date_range(start=date_range[0], end=date_range[1], freq='D')
    volumes = np.random.randint(50, 200, size=len(dates))
    chart_data = pd.DataFrame({
        'date': dates,
        'volume': volumes
    })
    fig = px.line(chart_data, x='date', y='volume', title='Daily Transaction Volume')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Anomaly Distribution")
    # Create sample anomaly data
    scores = np.linspace(0.1, 1.0, 10)
    counts = np.random.randint(1, 100, size=len(scores))
    anomaly_data = pd.DataFrame({
        'score': scores,
        'count': counts
    })
    fig = px.histogram(anomaly_data, x='score', y='count', 
                      title='Distribution of Anomaly Scores',
                      labels={'score': 'Anomaly Score', 'count': 'Number of Transactions'})
    st.plotly_chart(fig, use_container_width=True)

# Recent Anomalies
st.subheader("Recent Anomalies")
# Create sample anomaly records
sample_dates = pd.date_range(start=date_range[0], periods=7, freq='D')
sample_hashes = [f'0x{i:064x}' for i in range(7)]  # Generate proper length hashes
sample_scores = np.random.uniform(0.85, 0.99, size=7)

anomalies = pd.DataFrame({
    'timestamp': sample_dates,
    'transaction_hash': sample_hashes,
    'anomaly_score': sample_scores
}).sort_values('anomaly_score', ascending=False)

# Format the display
st.dataframe(
    anomalies.style.format({
        'timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M:%S'),
        'anomaly_score': '{:.4f}'
    })
)
