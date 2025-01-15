import pandas as pd
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
    results_if = anomaly_detection_pipeline(df, model_type="isolation_forest")
    n_anomalies_if = results_if['is_anomaly'].sum()
    print(f"Isolation Forest detected {n_anomalies_if} anomalies")
    
    # DBSCAN
    results_dbscan = anomaly_detection_pipeline(df, model_type="dbscan")
    n_anomalies_dbscan = results_dbscan['is_anomaly'].sum()
    print(f"DBSCAN detected {n_anomalies_dbscan} anomalies")
    
    # 2. Tune Isolation Forest
    print("\n2. Tuning Isolation Forest...")
    best_if_params = tune_isolation_forest(df)
    print(f"Best Isolation Forest parameters: {best_if_params}")
    
    # Run anomaly detection with tuned parameters
    results_if_tuned = anomaly_detection_pipeline(
        df,
        model_type="isolation_forest",
        params=best_if_params
    )
    n_anomalies_if_tuned = results_if_tuned['is_anomaly'].sum()
    print(f"Tuned Isolation Forest detected {n_anomalies_if_tuned} anomalies")
    
    # 3. Tune DBSCAN
    print("\n3. Tuning DBSCAN...")
    best_dbscan_params = tune_dbscan(df)
    print(f"Best DBSCAN parameters: {best_dbscan_params}")
    
    # Run anomaly detection with tuned parameters
    results_dbscan_tuned = anomaly_detection_pipeline(
        df,
        model_type="dbscan",
        params=best_dbscan_params
    )
    n_anomalies_dbscan_tuned = results_dbscan_tuned['is_anomaly'].sum()
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