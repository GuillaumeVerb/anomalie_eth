import pandas as pd
import numpy as np
import mlflow
from pathlib import Path

from src.modeling import (
    anomaly_detection_pipeline,
    tune_isolation_forest,
    tune_dbscan
)
from src.config import PROCESSED_DATA_DIR, EXPERIMENT_NAME

def main():
    # Load processed data
    data_file = PROCESSED_DATA_DIR / "transactions_processed.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Please ensure {data_file} exists before running this example")
    
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} transactions from {data_file}")
    
    # 1. Run basic anomaly detection with both models
    print("\n1. Running basic anomaly detection...")
    
    # Isolation Forest
    predictions_if, model_if = anomaly_detection_pipeline(df, model_type="isolation_forest")
    n_anomalies_if = np.sum(predictions_if == 1)
    print(f"Isolation Forest detected {n_anomalies_if} anomalies")
    
    # DBSCAN
    predictions_dbscan, model_dbscan = anomaly_detection_pipeline(df, model_type="dbscan")
    n_anomalies_dbscan = np.sum(predictions_dbscan == 1)
    print(f"DBSCAN detected {n_anomalies_dbscan} anomalies")
    
    # 2. Tune Isolation Forest
    print("\n2. Tuning Isolation Forest...")
    param_grid_if = {
        'n_estimators': [50, 100, 200],
        'contamination': [0.01, 0.05, 0.1],
        'max_samples': ['auto', 100, 500]
    }
    best_if_params = tune_isolation_forest(df, param_grid_if)
    print(f"Best Isolation Forest parameters: {best_if_params}")
    
    # Run anomaly detection with tuned parameters
    predictions_if_tuned, model_if_tuned = anomaly_detection_pipeline(
        df,
        model_type="isolation_forest"
    )
    n_anomalies_if_tuned = np.sum(predictions_if_tuned == 1)
    print(f"Tuned Isolation Forest detected {n_anomalies_if_tuned} anomalies")
    
    # 3. Tune DBSCAN
    print("\n3. Tuning DBSCAN...")
    param_grid_dbscan = {
        'eps': [0.1, 0.5, 1.0, 2.0],
        'min_samples': [2, 5, 10]
    }
    best_dbscan_params = tune_dbscan(df, param_grid_dbscan)
    print(f"Best DBSCAN parameters: {best_dbscan_params}")
    
    # Run anomaly detection with tuned parameters
    predictions_dbscan_tuned, model_dbscan_tuned = anomaly_detection_pipeline(
        df,
        model_type="dbscan"
    )
    n_anomalies_dbscan_tuned = np.sum(predictions_dbscan_tuned == 1)
    print(f"Tuned DBSCAN detected {n_anomalies_dbscan_tuned} anomalies")
    
    print("\nAll results are logged in MLflow")
    print(f"To view results, run: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("Then open http://localhost:5000 in your browser")

if __name__ == "__main__":
    # Ensure the examples directory exists
    Path("examples").mkdir(exist_ok=True)
    
    # Set MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    main() 