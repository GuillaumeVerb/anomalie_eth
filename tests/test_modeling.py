import pandas as pd
import numpy as np
from src.modeling import anomaly_detection_pipeline

def test_anomaly_detection_pipeline():
    # Create a sample DataFrame with required features
    sample_data = {
        'value': [1.0, 2.0, 3.0, 100.0],  # Last value is an outlier
        'gas_price': [0.1, 0.2, 0.3, 0.2],
        'gas': [21000, 21000, 21000, 21000],
        'transaction_fee': [0.1, 0.2, 0.3, 0.2],
        'log_value': [0.0, 0.69, 1.09, 4.60],
        'log_gas_price': [-2.3, -1.6, -1.2, -1.6],
        'log_gas': [9.95, 9.95, 9.95, 9.95],
        'log_transaction_fee': [-2.3, -1.6, -1.2, -1.6],
        'from_address_freq': [2, 1, 1, 1],
        'to_address_freq': [2, 2, 1, 1],
        'value_to_gas_ratio': [0.05, 0.1, 0.15, 5.0],
        'fee_to_value_ratio': [0.1, 0.1, 0.1, 0.002]
    }
    df = pd.DataFrame(sample_data)
    
    # Run anomaly detection
    results = anomaly_detection_pipeline(df, model_type="IF")
    
    # Check if results are as expected
    assert isinstance(results, pd.DataFrame)
    assert 'anomaly' in results.columns
    assert results['anomaly'].dtype == bool
    assert results['anomaly'].sum() > 0  # At least one anomaly should be detected
    assert results['anomaly'].iloc[3]  # The last row should be detected as an anomaly 