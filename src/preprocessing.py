import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime

from src.config import (
    PROCESSED_DATA_DIR,
    FEATURES,
    LOG_LEVEL,
    LOG_FORMAT
)

# Logging configuration
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def calculate_address_frequencies(df):
    """
    Calculate frequency of appearance for from/to addresses.
    """
    from_freq = df['from_address'].value_counts().to_dict()
    to_freq = df['to_address'].value_counts().to_dict()
    
    # Add frequencies as new columns
    df['from_address_freq'] = df['from_address'].map(from_freq)
    df['to_address_freq'] = df['to_address'].map(to_freq)
    
    return df

def add_temporal_features(df):
    """
    Add time-based features from block_timestamp.
    """
    df['timestamp'] = pd.to_datetime(df['block_timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Calculate transaction density (transactions per minute in the block)
    df['tx_density'] = df.groupby('block_number')['timestamp'].transform('count')
    
    return df

def add_derived_features(df):
    """
    Add derived features from existing columns.
    """
    # Log transformations for numerical columns
    df['log_value'] = np.log1p(df['value'])
    df['log_gas_price'] = np.log1p(df['gas_price'])
    df['log_gas_used'] = np.log1p(df['gas_used'])
    df['log_transaction_fee'] = np.log1p(df['transaction_fee'])
    
    # Ratio features
    df['value_to_gas_ratio'] = df['value'] / (df['gas_used'] + 1)  # Add 1 to avoid division by zero
    df['fee_to_value_ratio'] = df['transaction_fee'] / (df['value'] + 1e-10)
    
    # Contract interaction flag
    df['is_contract_creation'] = df['to_address'].isna().astype(int)
    df['has_input_data'] = df['input_data'].notna().astype(int)
    
    return df

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    """
    # Fill missing 'to_address' for contract creations
    df['to_address'] = df['to_address'].fillna('CONTRACT_CREATION')
    
    # Fill missing numerical values with 0
    numerical_columns = ['value', 'gas_price', 'gas_used', 'transaction_fee']
    df[numerical_columns] = df[numerical_columns].fillna(0)
    
    # Fill missing input_data with empty string
    df['input_data'] = df['input_data'].fillna('')
    
    return df

def transform_data(df):
    """
    Transform raw transaction data by adding features and handling missing values.
    
    Args:
        df (pd.DataFrame): Raw transaction data
    
    Returns:
        pd.DataFrame: Transformed data with additional features
    """
    try:
        logger.info("Starting data transformation...")
        
        # Convert numerical columns to float
        numerical_columns = ['value', 'gas_price', 'gas_used', 'transaction_fee']
        for col in numerical_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Apply transformations
        df = handle_missing_values(df)
        df = calculate_address_frequencies(df)
        df = add_temporal_features(df)
        df = add_derived_features(df)
        
        # Remove any remaining NaN values
        df = df.fillna(0)
        
        # Select and order columns
        feature_columns = FEATURES + [
            'log_value', 'log_gas_price', 'log_gas_used', 'log_transaction_fee',
            'from_address_freq', 'to_address_freq',
            'value_to_gas_ratio', 'fee_to_value_ratio',
            'is_contract_creation', 'has_input_data',
            'hour', 'day_of_week', 'tx_density'
        ]
        
        # Ensure all required columns exist
        available_columns = [col for col in feature_columns if col in df.columns]
        
        logger.info(f"Transformation completed. Shape: {df.shape}")
        return df[available_columns]

    except Exception as e:
        logger.error(f"Error during data transformation: {str(e)}")
        raise

def save_processed_data(df, filename=None):
    """
    Save processed data to CSV file.
    
    Args:
        df (pd.DataFrame): Processed transaction data
        filename (str, optional): Custom filename. If None, timestamp will be used.
    
    Returns:
        Path: Path to the saved file
    """
    try:
        # Create processed data directory if it doesn't exist
        Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transactions_processed_{timestamp}.csv"
        
        output_path = Path(PROCESSED_DATA_DIR) / filename
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        
        return output_path

    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Example usage
        from src.data_collection import fetch_transactions
        
        # Fetch raw data
        raw_data_path = fetch_transactions()
        raw_df = pd.read_csv(raw_data_path)
        
        # Transform data
        processed_df = transform_data(raw_df)
        
        # Save processed data
        save_processed_data(processed_df)
        
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        sys.exit(1)
