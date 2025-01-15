import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Set up test environment
os.environ.setdefault('ETHERSCAN_API_KEY', 'mock_key')
os.environ.setdefault('ETH_NODE_URL', 'https://mainnet.infura.io/v3/mock-project-id')

# Create necessary directories
for dir_path in ['data/raw', 'data/processed']:
    Path(dir_path).mkdir(parents=True, exist_ok=True) 