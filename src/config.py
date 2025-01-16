import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import sys

# Load environment variables
load_dotenv()

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# MLflow configuration
if os.getenv("PYTEST_CURRENT_TEST"):
    # Use in-memory SQLite for tests
    MLFLOW_TRACKING_URI = "sqlite:///:memory:"
    # Set MLflow logging level to DEBUG during tests
    mlflow.set_logging_level(logging.DEBUG)
else:
    # Use file-based SQLite for development
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ethereum_anomaly_detection")

# Ethereum node configuration
ETH_NODE_URL = os.getenv("ETH_NODE_URL", "https://mainnet.infura.io/v3/your-project-id")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "your-api-key")

# Logging configuration
LOG_LEVEL = logging.DEBUG if os.getenv("PYTEST_CURRENT_TEST") else logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)

# Remove any existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add handlers
if os.getenv("PYTEST_CURRENT_TEST"):
    # During tests, log to console with more details
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    ))
    root_logger.addHandler(console_handler)
else:
    # Normal operation: log to file and console
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

# Get logger for this module
logger = logging.getLogger(__name__)
logger.debug("Logging configured with level: %s", LOG_LEVEL)

# Model parameters
ISOLATION_FOREST_PARAMS = {
    'n_estimators': 100,
    'contamination': 0.1,
    'random_state': 42
}

DBSCAN_PARAMS = {
    'eps': 0.5,
    'min_samples': 5
}

# Feature columns for anomaly detection
FEATURE_COLUMNS = ['value', 'gas_price', 'gas', 'transaction_fee'] 