import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import mlflow

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
# To start MLflow UI locally, run: mlflow ui --backend-store-uri sqlite:///mlflow.db
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "ethereum_anomaly_detection"

# Ethereum node configuration
ETH_NODE_URL = os.getenv("ETH_NODE_URL", "https://mainnet.infura.io/v3/your-project-id")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "your-api-key")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
FEATURE_COLUMNS = ['value', 'gas_price', 'gas_used'] 