# Ethereum Transaction Anomaly Detection

## 📋 Overview
This project implements a machine learning pipeline to detect anomalous transactions on the Ethereum blockchain. It uses historical transaction data to identify patterns that deviate from normal behavior, which could indicate potential suspicious activities.

## 🌟 Features
- **Data Collection**: Automated fetching of Ethereum transactions using Etherscan API
- **Data Preprocessing**: Comprehensive transaction data cleaning and feature engineering
- **Anomaly Detection**: Implementation of multiple detection algorithms:
  - Isolation Forest
  - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- **MLflow Integration**: Experiment tracking and model versioning
- **Interactive Dashboard**: Streamlit-based visualization of results
- **Docker Support**: Containerized application for easy deployment

## 🛠️ Technical Stack
- **Python 3.9**
- **Core Libraries**:
  - `web3`: Ethereum blockchain interaction
  - `pandas`: Data manipulation
  - `scikit-learn`: Machine learning algorithms
  - `mlflow`: ML experiment tracking
  - `streamlit`: Dashboard creation
- **Infrastructure**:
  - Docker
  - GitHub Actions (CI/CD)
  - MLflow server
  - SQLite database

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Docker (optional)
- Etherscan API key
- Ethereum Node URL (Infura or other provider)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/GuillaumeVerb/anomalie_eth.git
cd anomalie_eth
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file with your credentials:
```bash
ETH_NODE_URL=your_ethereum_node_url
ETHERSCAN_API_KEY=your_etherscan_api_key
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

### Docker Setup

Build and run using Docker Compose:
```bash
docker-compose up --build
```

This will start:
- MLflow server on port 5000
- Streamlit dashboard on port 8501

## 📊 Project Structure
```
anomalie_eth/
├── data/
│   ├── raw/            # Raw transaction data
│   └── processed/      # Preprocessed datasets
├── src/
│   ├── config.py       # Configuration parameters
│   ├── data_collection.py  # Ethereum data collection
│   ├── preprocessing.py    # Data cleaning and preprocessing
│   ├── modeling.py        # Anomaly detection models
│   └── etherscan_api.py   # Etherscan API wrapper
├── notebooks/
│   ├── 01_data_collection.ipynb  # Data collection exploration
│   └── 02_modeling.ipynb         # Model development
├── tests/
│   ├── test_preprocessing.py
│   └── test_modeling.py
├── dashboard/
│   └── app.py          # Streamlit dashboard
├── mlruns/            # MLflow artifacts
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 💻 Usage

### Data Collection
```python
from src.data_collection import collect_transactions

# Collect recent transactions
transactions = collect_transactions(
    start_block=12000000,
    end_block=12001000
)
```

### Model Training
```python
from src.modeling import train_anomaly_detector

# Train an anomaly detection model
model = train_anomaly_detector(
    data,
    algorithm="isolation_forest",
    experiment_id="your_experiment_id"
)
```

### Running Tests
```bash
python -m pytest tests/
```

## 📈 Dashboard Features
- Transaction volume visualization
- Anomaly score distribution
- Interactive transaction explorer
- Model performance metrics
- Real-time anomaly detection

## 🔄 CI/CD Pipeline
The project includes a GitHub Actions workflow that:
1. Runs all tests
2. Builds the Docker image
3. Pushes the image to Docker Hub
4. Deploys the application (if configured)

## 📝 Model Tracking
MLflow is used to track:
- Model parameters
- Performance metrics
- Model artifacts
- Experiment history

Access the MLflow UI at `http://localhost:5000` when running locally.

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 🔗 Links
- [Documentation](https://github.com/GuillaumeVerb/anomalie_eth/wiki)
- [Issue Tracker](https://github.com/GuillaumeVerb/anomalie_eth/issues)
- [Docker Hub Repository](https://hub.docker.com/r/guillaumeverb/guillaume_repo)

## 👥 Authors
- Guillaume Verbiguié
  - [LinkedIn](https://www.linkedin.com/in/guillaume-v-4832401b4/)
  - [Medium](https://guillaume-verbiguie.medium.com/)

## 🙏 Acknowledgments
- Etherscan API for providing transaction data
- The Ethereum community for blockchain insights
- Open-source ML community for algorithms and tools
