import pandas as pd
import numpy as np
import os
from pathlib import Path
from src.preprocessing import preprocess_transactions
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def test_preprocess_transactions():
    # Ensure directories exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create a sample DataFrame
    sample_data = {
        'value': [1.0, 2.0, 3.0],
        'gas_price': [0.1, 0.2, 0.3],
        'gas_used': [21000, 21000, 21000],
        'transaction_fee': [0.1, 0.2, 0.3],
        'from_address': ['0x123', '0x456', '0x123'],
        'to_address': ['0x789', '0x789', '0xabc'],
        'block_timestamp': [1704067200, 1704153600, 1704240000],  # Unix timestamps
        'input_data': ['', 'data', '']
    }
    df = pd.DataFrame(sample_data)
    
    test_file = RAW_DATA_DIR / 'test_transactions.csv'
    
    try:
        # Save sample data to a temporary CSV file
        df.to_csv(test_file, index=False)
        
        # Transform the data
        transformed_df = preprocess_transactions(test_file)
        
        # Check if required columns exist
        assert 'log_value' in transformed_df.columns
        assert 'log_gas_price' in transformed_df.columns
        assert 'from_address_freq' in transformed_df.columns
        assert 'to_address_freq' in transformed_df.columns
        
        # Check frequency calculations
        assert transformed_df['from_address_freq'].iloc[0] == 2  # '0x123' appears twice
        assert transformed_df['to_address_freq'].iloc[0] == 2  # '0x789' appears twice
        
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()
