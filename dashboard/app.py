import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np
from src.modeling import anomaly_detection_pipeline
from src.preprocessing import preprocess_transactions

# Configuration de la page
st.set_page_config(
    page_title="Ethereum Transaction Anomaly Detection",
    page_icon="📊",
    layout="wide"
)

# Fonctions avec cache
@st.cache_data
def load_data():
    """Load and cache the processed transactions data."""
    try:
        # Try to load processed data first
        data_path = Path("data/processed/transactions_processed.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
        else:
            # If not found, load raw data and process it
            raw_data_path = max(Path("data/raw").glob("transactions_*.csv"))
            df = preprocess_transactions(raw_data_path)
        
        # Add ETH conversions (1 ETH = 10^18 Wei)
        df['value_eth'] = df['value'] / 1e18
        df['transaction_fee_eth'] = df['transaction_fee'] / 1e18
        df['gas_price_gwei'] = df['gas_price'] / 1e9  # Convert to Gwei for better readability
        
        # Convert Unix timestamp to datetime and sort by timestamp
        df['timestamp'] = pd.to_datetime(df['block_timestamp'], unit='s')
        df = df.sort_values('timestamp')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def detect_anomalies(_df, model_type):
    """Run anomaly detection with selected model."""
    try:
        results = anomaly_detection_pipeline(_df, model_type)
        return results
    except Exception as e:
        st.error(f"Error in anomaly detection: {str(e)}")
        return None

def main():
    st.title("Ethereum Transaction Anomaly Detection")
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check the data files.")
        return
    
    # Get the actual date range from the data
    data_min_date = df['timestamp'].min()
    data_max_date = df['timestamp'].max()
    
    # Display data range info
    st.write(f"Data range: {data_min_date.strftime('%Y-%m-%d %H:%M:%S')} to {data_max_date.strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"Number of transactions: {len(df):,}")
    
    # Check if data spans multiple days
    unique_dates = df['timestamp'].dt.date.unique()
    if len(unique_dates) == 1:
        st.info(f"📅 Data is only available for {unique_dates[0].strftime('%Y-%m-%d')}. You can select different time ranges within this day.")
    
    # Date and time selection
    st.sidebar.header("Time Range Selection")
    
    if len(unique_dates) == 1:
        st.sidebar.markdown(f"**Date**: {unique_dates[0].strftime('%Y-%m-%d')} *(only available date)*")
        selected_date = unique_dates[0]
    else:
        selected_date = st.sidebar.date_input(
            "Date",
            value=data_min_date.date(),
            min_value=data_min_date.date(),
            max_value=data_max_date.date()
        )
    
    st.sidebar.subheader("Select Time Range")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_time = st.time_input(
            "Start Time",
            value=data_min_date.time()
        )
    
    with col2:
        end_time = st.time_input(
            "End Time",
            value=data_max_date.time()
        )
    
    # Create datetime objects for filtering
    start_datetime = pd.Timestamp.combine(selected_date, start_time)
    end_datetime = pd.Timestamp.combine(selected_date, end_time)
    
    # Handle case where end time is before start time
    if end_datetime < start_datetime:
        st.sidebar.error("⚠️ End time must be after start time")
        return
    
    # Filter data based on time range
    mask = (df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)
    filtered_df = df[mask].copy()
    
    # Show time range info with transaction count
    n_transactions = len(filtered_df)
    if n_transactions > 0:
        st.sidebar.success(f"✅ Selected {n_transactions:,} transactions from {start_datetime.strftime('%H:%M')} to {end_datetime.strftime('%H:%M')}")
    else:
        st.sidebar.warning("⚠️ No transactions found in the selected time range")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["IF", "DBSCAN"],
        format_func=lambda x: "Isolation Forest" if x == "IF" else x
    )
    
    # Select features for visualization
    st.sidebar.header("Visualization Settings")
    available_features = [
        'value_eth', 'value', 'gas_price_gwei', 'gas_price', 'gas_used', 
        'transaction_fee_eth', 'transaction_fee',
        'log_value', 'log_gas_price', 'log_gas_used', 'log_transaction_fee',
        'value_to_gas_ratio', 'fee_to_value_ratio'
    ]
    
    # Add tooltips for features
    feature_descriptions = {
        'value_eth': 'Transaction value in ETH',
        'value': 'Transaction value in Wei (1 ETH = 10^18 Wei)',
        'gas_price_gwei': 'Gas price in Gwei (1 Gwei = 10^9 Wei)',
        'gas_price': 'Gas price in Wei',
        'gas_used': 'Amount of gas used by the transaction',
        'transaction_fee_eth': 'Transaction fee in ETH',
        'transaction_fee': 'Transaction fee in Wei',
        'log_value': 'Natural log of value (for better scaling)',
        'log_gas_price': 'Natural log of gas price',
        'log_gas_used': 'Natural log of gas used',
        'log_transaction_fee': 'Natural log of transaction fee',
        'value_to_gas_ratio': 'Ratio of value to gas used',
        'fee_to_value_ratio': 'Ratio of transaction fee to value'
    }
    
    x_axis = st.sidebar.selectbox(
        "X-axis", 
        available_features, 
        index=0,
        help=feature_descriptions.get('value_eth', '')
    )
    
    y_axis = st.sidebar.selectbox(
        "Y-axis", 
        available_features, 
        index=5,
        help=feature_descriptions.get('transaction_fee_eth', '')
    )
    
    # Launch detection button
    if st.button("Launch Detection"):
        if len(filtered_df) == 0:
            st.warning("No transactions found in the selected time range.")
            return
            
        with st.spinner("Running anomaly detection..."):
            results = detect_anomalies(filtered_df, model_type)
            
            if results is not None:
                # Display metrics
                col1, col2, col3 = st.columns(3)
                total_transactions = len(results)
                anomaly_count = results['anomaly_label'].sum()
                anomaly_percentage = (anomaly_count / total_transactions) * 100
                
                col1.metric("Total Transactions", f"{total_transactions:,}")
                col2.metric("Detected Anomalies", f"{anomaly_count:,}")
                col3.metric("Anomaly Percentage", f"{anomaly_percentage:.2f}%")
                
                # Create scatter plot
                fig = px.scatter(
                    results,
                    x=x_axis,
                    y=y_axis,
                    color='anomaly_label',
                    color_discrete_map={0: 'blue', 1: 'red'},
                    title=f"Transaction Analysis: {x_axis} vs {y_axis}",
                    labels={'anomaly_label': 'Is Anomaly'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display anomalous transactions
                st.header("Anomalous Transactions")
                anomalies = results[results['anomaly_label'] == 1]
                
                display_cols = [
                    'value_eth', 'gas_price_gwei', 'gas_used', 'transaction_fee_eth',
                    'value_to_gas_ratio', 'fee_to_value_ratio',
                    'is_contract_creation', 'has_input_data'
                ]
                
                st.dataframe(
                    anomalies[display_cols].style.format({
                        'value_eth': '{:,.6f}',
                        'gas_price_gwei': '{:,.2f}',
                        'gas_used': '{:,.0f}',
                        'transaction_fee_eth': '{:,.6f}',
                        'value_to_gas_ratio': '{:,.4f}',
                        'fee_to_value_ratio': '{:,.4f}'
                    })
                )
                
                # Add download button for anomalies
                csv = anomalies[display_cols].to_csv(index=False)
                st.download_button(
                    "Download Anomalies CSV",
                    csv,
                    "anomalies.csv",
                    "text/csv",
                    key='download-csv'
                )

if __name__ == "__main__":
    main()
