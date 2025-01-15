import os
import sys
import shutil
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import mlflow

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import after adding to path
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, EXPERIMENT_NAME

# Set up test environment
os.environ.setdefault('ETHERSCAN_API_KEY', 'mock_key')
os.environ.setdefault('ETH_NODE_URL', 'https://mainnet.infura.io/v3/mock-project-id')

@pytest.fixture(autouse=True)
def setup_test_env(tmp_path):
    """Set up test environment before each test."""
    # Create necessary directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set up MLflow to use a temporary directory
    mlflow_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    
    # Create test experiment
    mlflow.create_experiment(EXPERIMENT_NAME)
    
    # Create empty test file
    test_file = RAW_DATA_DIR / 'transactions_test.csv'
    test_file.touch()
    
    yield
    
    # Cleanup after tests
    if test_file.exists():
        test_file.unlink()
    
    # Clean up MLflow directory
    if mlflow_dir.exists():
        shutil.rmtree(mlflow_dir)
    
    # Reset MLflow
    mlflow.end_run()

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