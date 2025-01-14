import pandas as pd
import numpy as np
from src.preprocessing import transform_data

def test_transform_data():
    # Create a sample DataFrame
    sample_data = {
        'value': [1.0, 2.0, 3.0],
        'gas_price': [0.1, 0.2, 0.3],
        'gas_used': [21000, 21000, 21000],
        'transaction_fee': [0.1, 0.2, 0.3],
        'from_address': ['0x123', '0x456', '0x123'],
        'to_address': ['0x789', '0x789', '0xabc'],
        'block_timestamp': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'input_data': ['', 'data', '']
    }
    df = pd.DataFrame(sample_data)
    
    # Transform the data
    transformed_df = transform_data(df)
    
    # Check if required columns exist
    assert 'log_value' in transformed_df.columns
    assert 'log_gas_price' in transformed_df.columns
    assert 'from_address_freq' in transformed_df.columns
    assert 'to_address_freq' in transformed_df.columns
    
    # Check frequency calculations
    assert transformed_df['from_address_freq'].iloc[0] == 2  # '0x123' appears twice
    assert transformed_df['to_address_freq'].iloc[0] == 2  # '0x789' appears twice
