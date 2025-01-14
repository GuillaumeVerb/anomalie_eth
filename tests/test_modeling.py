import pandas as pd
import numpy as np
from src.modeling import anomaly_detection_pipeline

def test_anomaly_detection_pipeline():
    # Create a sample DataFrame with required features
    sample_data = {
        'value': [1.0, 2.0, 3.0, 100.0],  # Last value is an outlier
        'gas_price': [0.1, 0.2, 0.3, 0.2],
        'gas_used': [21000, 21000, 21000, 21000],
        'transaction_fee': [0.1, 0.2, 0.3, 0.2],
        'log_value': [0.0, 0.69, 1.09, 4.60],
        'log_gas_price': [-2.3, -1.6, -1.2, -1.6],
        'log_gas_used': [9.95, 9.95, 9.95, 9.95],
        'log_transaction_fee': [-2.3, -1.6, -1.2, -1.6],
        'from_address_freq': [2, 1, 1, 1],
        'to_address_freq': [2, 2, 1, 1],
        'value_to_gas_ratio': [0.05, 0.1, 0.15, 5.0],
        'fee_to_value_ratio': [0.1, 0.1, 0.1, 0.002]
    }
    df = pd.DataFrame(sample_data)
    
    # Run anomaly detection
    results = anomaly_detection_pipeline(df, model_type="IF")
    
    # Check if anomaly_label column was added
    assert 'anomaly_label' in results.columns
    
    # Check if we have binary labels (0 or 1)
    assert set(results['anomaly_label'].unique()).issubset({0, 1})
    
    # Check if at least one anomaly was detected (the outlier)
    assert results['anomaly_label'].sum() > 0 