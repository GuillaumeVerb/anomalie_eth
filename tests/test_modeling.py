import pandas as pd
import numpy as np
import pytest
from src.modeling import anomaly_detection_pipeline

@pytest.fixture
def sample_data():
    """Create a sample DataFrame with required features."""
    return pd.DataFrame({
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
    })

@pytest.mark.parametrize("model_type", ["IF", "DBSCAN"])
def test_anomaly_detection_pipeline(sample_data, model_type):
    """Test anomaly detection with different models."""
    # Run anomaly detection
    results = anomaly_detection_pipeline(sample_data, model_type)
    
    # Basic checks
    assert isinstance(results, pd.DataFrame)
    assert 'anomaly_label' in results.columns
    
    # Convert to boolean for consistent comparison
    anomalies = results['anomaly_label'] == 1
    
    # At least one anomaly should be detected (we have an outlier)
    assert anomalies.sum() > 0
    
    # The last row should be detected as an anomaly (it's an obvious outlier)
    assert anomalies.iloc[3]
    
    # Check that we didn't lose any data
    assert len(results) == len(sample_data)
    
    # Check that all original columns are preserved
    for col in sample_data.columns:
        assert col in results.columns 