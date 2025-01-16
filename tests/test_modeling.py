import pandas as pd
import numpy as np
import pytest
from src.modeling import anomaly_detection_pipeline
from src.preprocessing import preprocess_transactions
from src.config import FEATURE_COLUMNS, RAW_DATA_DIR, EXPERIMENT_NAME
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
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"Starting test for model type: {model_type}")
    logger.info("-" * 80)
    
    # Log MLflow configuration
    logger.debug(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    experiment = mlflow.get_experiment(setup_mlflow)
    logger.debug(f"Using experiment: {experiment.name} (ID: {experiment.experiment_id})")
    
    # Make sure no runs are active
    try:
        active_run = mlflow.active_run()
        if active_run:
            logger.warning(f"Found active run at test start: {active_run.info.run_id}")
            mlflow.end_run()
            logger.info("Ended active run")
    except Exception as e:
        logger.error(f"Error checking active run before test: {str(e)}", exc_info=True)
    
    try:
        # Log input data summary
        logger.debug("Input data summary:")
        logger.debug(f"Shape: {sample_data.shape}")
        logger.debug(f"Columns: {sample_data.columns.tolist()}")
        logger.debug(f"Feature columns: {FEATURE_COLUMNS}")
        
        # Run anomaly detection
        logger.info("Running anomaly detection pipeline")
        results = anomaly_detection_pipeline(sample_data, model_type)
        logger.info("Anomaly detection completed")
        
        # Log results summary
        logger.debug("Results summary:")
        logger.debug(f"Shape: {results.shape}")
        logger.debug(f"Columns: {results.columns.tolist()}")
        
        # Basic checks
        assert isinstance(results, pd.DataFrame)
        assert 'anomaly_label' in results.columns
        
        # Convert to boolean for consistent comparison
        anomalies = results['anomaly_label'] == 1
        n_anomalies = anomalies.sum()
        logger.info(f"Detected {n_anomalies} anomalies ({n_anomalies/len(results)*100:.1f}%)")
        
        # Log anomaly details
        anomaly_indices = anomalies[anomalies].index.tolist()
        logger.debug(f"Anomaly indices: {anomaly_indices}")
        
        # At least one anomaly should be detected (we have outliers)
        assert n_anomalies > 0
        
        # The last row should be detected as an anomaly (it's an obvious outlier in all features)
        assert anomalies.iloc[3]
        
        # Check that we didn't lose any data
        assert len(results) == len(sample_data)
        
        # Check that all required feature columns exist
        for col in FEATURE_COLUMNS:
            assert col in results.columns, f"Required feature column {col} is missing"
            
        logger.info("All test assertions passed")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Make sure no runs are left active
        try:
            active_run = mlflow.active_run()
            if active_run:
                logger.warning(f"Found active run at test end: {active_run.info.run_id}")
                mlflow.end_run()
                logger.info("Ended active run")
        except Exception as e:
            logger.error(f"Error ending active run after test: {str(e)}", exc_info=True)
        logger.info("-" * 80)
        logger.info(f"Test completed for model type: {model_type}")
        logger.info("=" * 80) 