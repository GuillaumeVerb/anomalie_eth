import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_COLUMNS,
    logger
)

def preprocess_transactions(input_file=None):
    """
    Preprocess Ethereum transactions for anomaly detection.
    
    Args:
        input_file: Path to the input CSV file. If None, uses the most recent file in RAW_DATA_DIR.
    
    Returns:
        Path to the processed CSV file.
    """
    try:
        # Find most recent file if not specified
        if input_file is None:
            files = list(RAW_DATA_DIR.glob("transactions_*.csv"))
            if not files:
                raise FileNotFoundError("No transaction files found in data/raw/")
            input_file = max(files, key=lambda x: x.stat().st_mtime)
        
        logger.info(f"Processing transactions from {input_file}")
        
        # Read data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} transactions")
        
        # Drop rows with missing values in key columns
        df = df.dropna(subset=['value', 'gas_price', 'gas_used'])
        logger.info(f"Remaining transactions after dropping nulls: {len(df)}")
        
        # Create derived features
        df['transaction_fee'] = df['gas_price'] * df['gas_used']
        
        # Log transform numerical features to handle extreme values
        for col in ['value', 'gas_price', 'gas_used', 'transaction_fee']:
            # Add small constant to handle zeros
            df[f'log_{col}'] = np.log1p(df[col])
        
        # Calculate transaction density (number of transactions per block)
        tx_counts = df['block_number'].value_counts()
        df['tx_density'] = df['block_number'].map(tx_counts)
        
        # Calculate address frequencies
        from_counts = df['from_address'].value_counts()
        to_counts = df['to_address'].value_counts()
        df['from_address_freq'] = df['from_address'].map(from_counts)
        df['to_address_freq'] = df['to_address'].map(to_counts)
        
        # Calculate ratios
        df['value_to_gas_ratio'] = df['value'] / df['gas']
        df['fee_to_value_ratio'] = df['transaction_fee'] / (df['value'] + 1e-10)  # Add small constant to avoid division by zero
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = PROCESSED_DATA_DIR / f"transactions_processed.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"Processed data saved to {output_file}")
        logger.info("Feature summary:")
        logger.info(df[FEATURE_COLUMNS].describe())
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    # Create processed data directory if it doesn't exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process transactions
    preprocess_transactions()
