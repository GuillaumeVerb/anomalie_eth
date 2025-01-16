import pandas as pd
import numpy as np
import pytest
from src.modeling import anomaly_detection_pipeline
from src.preprocessing import preprocess_transactions
from src.config import FEATURE_COLUMNS, RAW_DATA_DIR
import mlflow
import logging

@pytest.fixture
def sample_data(setup_test_env):
    """Create a sample DataFrame with required features."""
    np.random.seed(42)  # For reproducibility
    
    # Create sample data with clear anomalies in each feature
    df = pd.DataFrame({
        'value': [1.0, 1.2, 0.8, 100.0],  # Last value is an outlier
        'gas_price': [20, 22, 21, 200],    # Last value is an outlier
        'gas': [21000, 21000, 21000, 500000],  # Last value is an outlier
        'transaction_fee': [420000, 462000, 441000, 100000000],  # gas_price * gas
        'block_number': [1000, 1001, 1002, 1003],
        'from_address': ['0x123', '0x456', '0x789', '0x123'],
        'to_address': ['0x789', '0x789', '0xabc', '0xdef'],
        'input_data': ['', '', '', '0x123456'],  # Last one has contract interaction
        'block_timestamp': [1704067200, 1704153600, 1704240000, 1704326400]  # Daily intervals
    })
    
    # Save to CSV and preprocess
    test_file = RAW_DATA_DIR / 'transactions_test.csv'
    df.to_csv(test_file, index=False)
    
    # Process the data
    processed_df = preprocess_transactions(test_file)
    return processed_df

@pytest.mark.parametrize("model_type", ["IF", "DBSCAN"])
def test_anomaly_detection_pipeline(sample_data, model_type, setup_mlflow):
    """Test anomaly detection with different models."""
    logging.info(f"Starting test for model type: {model_type}")
    
    # Make sure no runs are active
    try:
        active_run = mlflow.active_run()
        if active_run:
            logging.info(f"Ending active run before test: {active_run.info.run_id}")
            mlflow.end_run()
    except Exception as e:
        logging.warning(f"Error checking active run before test: {str(e)}")
    
    try:
        # Run anomaly detection
        logging.info("Running anomaly detection pipeline")
        results = anomaly_detection_pipeline(sample_data, model_type)
        logging.info("Anomaly detection completed")
        
        # Basic checks
        assert isinstance(results, pd.DataFrame)
        assert 'anomaly_label' in results.columns
        
        # Convert to boolean for consistent comparison
        anomalies = results['anomaly_label'] == 1
        n_anomalies = anomalies.sum()
        logging.info(f"Detected {n_anomalies} anomalies")
        
        # At least one anomaly should be detected (we have outliers)
        assert n_anomalies > 0
        
        # The last row should be detected as an anomaly (it's an obvious outlier in all features)
        assert anomalies.iloc[3]
        
        # Check that we didn't lose any data
        assert len(results) == len(sample_data)
        
        # Check that all required feature columns exist
        for col in FEATURE_COLUMNS:
            assert col in results.columns, f"Required feature column {col} is missing"
            
        logging.info("All test assertions passed")
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        raise
    finally:
        # Make sure no runs are left active
        try:
            active_run = mlflow.active_run()
            if active_run:
                logging.info(f"Ending active run after test: {active_run.info.run_id}")
                mlflow.end_run()
        except Exception as e:
            logging.warning(f"Error ending active run after test: {str(e)}")
        logging.info("Test cleanup completed") 