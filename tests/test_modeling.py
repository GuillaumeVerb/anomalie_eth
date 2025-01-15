import pandas as pd
import numpy as np
import pytest
from src.modeling import anomaly_detection_pipeline
from src.preprocessing import preprocess_transactions
from src.config import FEATURE_COLUMNS, RAW_DATA_DIR

@pytest.fixture
def sample_data(setup_test_env):
    """Create a sample DataFrame with required features."""
    df = pd.DataFrame({
        'value': [1.0, 2.0, 3.0, 100.0],  # Last value is an outlier
        'gas_price': [0.1, 0.2, 0.3, 0.2],
        'gas': [21000, 21000, 21000, 21000],
        'block_number': [1000, 1001, 1002, 1003],
        'from_address': ['0x123', '0x456', '0x789', '0x123'],
        'to_address': ['0x789', '0x789', '0xabc', '0xdef'],
        'input_data': ['', '', '', ''],
        'block_timestamp': [1704067200, 1704153600, 1704240000, 1704326400]
    })
    
    # Save to CSV and preprocess
    test_file = RAW_DATA_DIR / 'transactions_test.csv'
    df.to_csv(test_file, index=False)
    
    # Process the data
    processed_df = preprocess_transactions(test_file)
    return processed_df

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
    
    # Check that all required feature columns exist
    for col in FEATURE_COLUMNS:
        assert col in results.columns, f"Required feature column {col} is missing" 