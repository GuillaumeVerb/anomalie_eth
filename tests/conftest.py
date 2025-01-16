import os
import sys
import shutil
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import mlflow
import tempfile
import logging

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import after adding to path
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, EXPERIMENT_NAME

# Set up test environment
os.environ.setdefault('ETHERSCAN_API_KEY', 'mock_key')
os.environ.setdefault('ETH_NODE_URL', 'https://mainnet.infura.io/v3/mock-project-id')

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture(autouse=True)
def setup_mlflow(tmp_path):
    """Set up MLflow tracking for tests."""
    # Set up MLflow to use a temporary directory
    mlflow_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    
    # End any active runs
    mlflow.end_run()
    
    # Delete existing experiment if it exists
    try:
        existing_exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if existing_exp:
            mlflow.delete_experiment(existing_exp.experiment_id)
    except Exception as e:
        logging.warning(f"Error cleaning up existing experiment: {str(e)}")
    
    # Create new experiment
    try:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        logging.info(f"Created test experiment with ID: {experiment_id}")
    except Exception as e:
        logging.error(f"Failed to create experiment: {str(e)}")
        raise
    
    yield
    
    # Cleanup
    try:
        mlflow.end_run()
        if mlflow_dir.exists():
            shutil.rmtree(mlflow_dir)
    except Exception as e:
        logging.warning(f"Error during MLflow cleanup: {str(e)}")

@pytest.fixture(autouse=True)
def setup_test_env(test_data_dir, setup_mlflow):
    """Set up test environment before each test."""
    # Create necessary directories in temp location
    raw_dir = test_data_dir / "raw"
    processed_dir = test_data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Override config directories for testing
    with patch('src.config.RAW_DATA_DIR', raw_dir), \
         patch('src.config.PROCESSED_DATA_DIR', processed_dir):
        
        # Create empty test file
        test_file = raw_dir / 'transactions_test.csv'
        test_file.touch()
        
        yield
        
        # Cleanup after tests
        if test_file.exists():
            test_file.unlink()

@pytest.fixture(autouse=True)
def mock_external_apis():
    """Mock all external API calls."""
    with patch('src.etherscan_api._make_request') as mock_etherscan:
        mock_etherscan.return_value = {
            'status': '1',
            'message': 'OK',
            'result': '1000000000000000000'  # 1 ETH in Wei
        }
        
        with patch('web3.Web3.HTTPProvider') as mock_web3_provider:
            mock_provider = MagicMock()
            mock_web3_provider.return_value = mock_provider
            
            yield 