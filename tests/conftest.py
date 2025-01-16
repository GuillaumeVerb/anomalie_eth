import os
import sys
import shutil
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import mlflow
import tempfile
import logging
import time

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

@pytest.fixture(scope="session")
def mlflow_tracking_uri(tmp_path_factory):
    """Create a session-wide MLflow tracking directory."""
    logging.info("Setting up MLflow tracking URI")
    # Use memory SQLite database for testing
    uri = "sqlite:///:memory:"
    mlflow.set_tracking_uri(uri)
    logging.info(f"Set MLflow tracking URI to: {uri}")
    
    # Make sure no runs are active at the start
    try:
        active_run = mlflow.active_run()
        if active_run:
            logging.info(f"Ending active run at start: {active_run.info.run_id}")
            mlflow.end_run()
    except Exception as e:
        logging.warning(f"Error checking active run at start: {str(e)}")
        
    yield uri
    
    # Cleanup at the end of the session
    logging.info("Cleaning up MLflow at session end")
    try:
        # End any active runs
        try:
            active_run = mlflow.active_run()
            if active_run:
                logging.info(f"Ending active run at cleanup: {active_run.info.run_id}")
                mlflow.end_run()
        except Exception as e:
            logging.warning(f"Error ending active run at cleanup: {str(e)}")
            
        # Delete all experiments
        experiments = mlflow.search_experiments()
        for exp in experiments:
            try:
                logging.info(f"Cleaning up experiment: {exp.experiment_id}")
                # End all runs in the experiment
                runs = mlflow.search_runs([exp.experiment_id])
                for _, run in runs.iterrows():
                    try:
                        run_info = mlflow.get_run(run.run_id)
                        if run_info.info.status != "FINISHED":
                            logging.info(f"Ending unfinished run: {run.run_id}")
                            mlflow.end_run(run.run_id)
                    except Exception as e:
                        logging.warning(f"Error ending run {run.run_id}: {str(e)}")
                # Delete the experiment
                logging.info(f"Deleting experiment: {exp.experiment_id}")
                mlflow.delete_experiment(exp.experiment_id)
            except Exception as e:
                logging.warning(f"Error cleaning up experiment {exp.experiment_id}: {str(e)}")
    except Exception as e:
        logging.error(f"Error during MLflow cleanup: {str(e)}")

@pytest.fixture(autouse=True)
def setup_mlflow(mlflow_tracking_uri):
    """Set up MLflow tracking for tests."""
    logging.info("Setting up MLflow for test")
    # Make sure we start with a clean state
    try:
        active_run = mlflow.active_run()
        if active_run:
            logging.info(f"Ending active run at test start: {active_run.info.run_id}")
            mlflow.end_run()
    except Exception as e:
        logging.warning(f"Error checking active run at test start: {str(e)}")
    
    # Delete existing experiment if it exists
    try:
        existing_exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if existing_exp:
            logging.info(f"Found existing experiment: {existing_exp.experiment_id}")
            # End all runs in the experiment
            runs = mlflow.search_runs([existing_exp.experiment_id])
            for _, run in runs.iterrows():
                try:
                    run_info = mlflow.get_run(run.run_id)
                    if run_info.info.status != "FINISHED":
                        logging.info(f"Ending unfinished run in existing experiment: {run.run_id}")
                        mlflow.end_run(run.run_id)
                except Exception as e:
                    logging.warning(f"Error ending run {run.run_id}: {str(e)}")
            # Delete the experiment
            logging.info(f"Deleting existing experiment: {existing_exp.experiment_id}")
            mlflow.delete_experiment(existing_exp.experiment_id)
    except Exception as e:
        logging.warning(f"Error handling existing experiment: {str(e)}")
    
    # Create new experiment
    try:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={
                "test": "true",
                "created_by": "pytest",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_run": str(time.time())
            }
        )
        logging.info(f"Created new test experiment with ID: {experiment_id}")
    except Exception as e:
        logging.error(f"Failed to create experiment: {str(e)}")
        raise
    
    yield experiment_id
    
    # Cleanup
    logging.info("Cleaning up MLflow after test")
    try:
        # End any active runs
        try:
            active_run = mlflow.active_run()
            if active_run:
                logging.info(f"Ending active run at test end: {active_run.info.run_id}")
                mlflow.end_run()
        except Exception as e:
            logging.warning(f"Error ending active run at test end: {str(e)}")
        
        # Delete all runs in the experiment
        try:
            runs = mlflow.search_runs([experiment_id])
            for _, run in runs.iterrows():
                try:
                    run_info = mlflow.get_run(run.run_id)
                    if run_info.info.status != "FINISHED":
                        logging.info(f"Ending unfinished run in cleanup: {run.run_id}")
                        mlflow.end_run(run.run_id)
                except Exception as e:
                    logging.warning(f"Error ending run {run.run_id}: {str(e)}")
        except Exception as e:
            logging.warning(f"Error cleaning up runs: {str(e)}")
        
        # Delete the experiment
        try:
            logging.info(f"Deleting test experiment: {experiment_id}")
            mlflow.delete_experiment(experiment_id)
        except Exception as e:
            logging.warning(f"Error deleting experiment: {str(e)}")
    except Exception as e:
        logging.error(f"Error during MLflow cleanup: {str(e)}")
    finally:
        # Make absolutely sure no runs are left active
        try:
            active_run = mlflow.active_run()
            if active_run:
                logging.info(f"Ending final active run: {active_run.info.run_id}")
                mlflow.end_run()
        except Exception as e:
            logging.warning(f"Error in final run cleanup: {str(e)}")

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

@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for test session."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add a console handler with detailed formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Log test session start
    logger.info("Starting test session")
    
    yield
    
    # Log test session end
    logger.info("Test session completed") 