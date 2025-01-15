import pandas as pd
import numpy as np
import os
from pathlib import Path
import pytest
from src.preprocessing import preprocess_transactions
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

@pytest.fixture
def sample_transactions():
    """Create a sample transactions DataFrame."""
    return pd.DataFrame({
        'value': [1.0, 2.0, 3.0],
        'gas_price': [0.1, 0.2, 0.3],
        'gas': [21000, 21000, 21000],
        'from_address': ['0x123', '0x456', '0x123'],
        'to_address': ['0x789', '0x789', '0xabc'],
        'block_timestamp': [1704067200, 1704153600, 1704240000],
        'input_data': ['', 'data', ''],
        'block_number': [1000, 1001, 1002]
    })

def test_preprocess_transactions(sample_transactions):
    """Test the preprocessing of transactions."""
    # Ensure directories exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    test_file = RAW_DATA_DIR / 'test_transactions.csv'
    
    try:
        # Save sample data
        sample_transactions.to_csv(test_file, index=False)
        
        # Transform the data
        transformed_df = preprocess_transactions(test_file)
        
        # Check if all derived columns exist
        expected_columns = [
            'transaction_fee',
            'log_value',
            'log_gas_price',
            'log_gas',
            'log_transaction_fee',
            'from_address_freq',
            'to_address_freq',
            'value_to_gas_ratio',
            'fee_to_value_ratio',
            'tx_density'
        ]
        
        for col in expected_columns:
            assert col in transformed_df.columns, f"Column {col} is missing"
        
        # Check frequency calculations
        assert transformed_df['from_address_freq'].iloc[0] == 2  # '0x123' appears twice
        assert transformed_df['to_address_freq'].iloc[0] == 2  # '0x789' appears twice
        
        # Check if derived values are calculated correctly
        assert np.allclose(
            transformed_df['transaction_fee'],
            sample_transactions['gas_price'] * sample_transactions['gas']
        )
        
        # Check log transformations
        assert np.allclose(
            transformed_df['log_value'],
            np.log1p(sample_transactions['value'])
        )
        
        # Check transaction density
        assert transformed_df['tx_density'].iloc[0] == 1  # One transaction per block
        
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()
