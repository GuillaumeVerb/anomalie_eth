import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import mlflow
import logging
from src.config import (
    FEATURE_COLUMNS,
    ISOLATION_FOREST_PARAMS,
    DBSCAN_PARAMS,
    EXPERIMENT_NAME
)

def anomaly_detection_pipeline(data: pd.DataFrame, model_type: str = "isolation_forest"):
    """
    Run anomaly detection pipeline on transaction data.
    
    Args:
        data (pd.DataFrame): Input transaction data
        model_type (str): Type of model to use ("isolation_forest" or "dbscan")
        
    Returns:
        pd.DataFrame: Original data with anomaly column added (True for anomalies)
    """
    # Extract features
    X = data[FEATURE_COLUMNS].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
    
    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Log dataset info
        mlflow.log_param("n_samples", len(data))
        mlflow.log_param("features", FEATURE_COLUMNS)
        
        if model_type.upper() in ["IF", "ISOLATION_FOREST"]:
            # Train Isolation Forest
            model = IsolationForest(**ISOLATION_FOREST_PARAMS)
            mlflow.log_params(ISOLATION_FOREST_PARAMS)
            
        elif model_type.upper() == "DBSCAN":
            # Train DBSCAN
            model = DBSCAN(**DBSCAN_PARAMS)
            mlflow.log_params(DBSCAN_PARAMS)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Fit model and get predictions
        predictions = model.fit_predict(X_scaled)
        
        # Convert predictions to boolean (True: anomaly, False: normal)
        if model_type.upper() in ["IF", "ISOLATION_FOREST"]:
            predictions = predictions == -1  # IF: 1 is inlier, -1 is outlier
        else:
            predictions = predictions == -1  # DBSCAN: -1 is outlier
            
        # Calculate silhouette score
        try:
            silhouette = silhouette_score(X_scaled, predictions)
            mlflow.log_metric("silhouette_score", silhouette)
            logging.info(f"Silhouette score: {silhouette:.3f}")
        except:
            logging.warning("Could not calculate silhouette score")
            
        # Log number of anomalies
        n_anomalies = np.sum(predictions)
        mlflow.log_metric("n_anomalies", n_anomalies)
        mlflow.log_metric("anomaly_ratio", n_anomalies / len(predictions))
        
        logging.info(f"Detected {n_anomalies} anomalies ({n_anomalies/len(predictions)*100:.1f}%)")
        
        # Add predictions to original data
        result_df = data.copy()
        result_df['anomaly'] = predictions
            
    return result_df

def tune_isolation_forest(data: pd.DataFrame, param_grid: dict):
    """
    Tune Isolation Forest hyperparameters using grid search.
    
    Args:
        data (pd.DataFrame): Input transaction data
        param_grid (dict): Parameter grid to search
        
    Returns:
        dict: Best parameters found
    """
    X = data[FEATURE_COLUMNS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
    
    best_score = -np.inf
    best_params = None
    
    # Grid search
    for params in _get_param_combinations(param_grid):
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.log_params(params)
            
            model = IsolationForest(**params)
            predictions = model.fit_predict(X_scaled)
            predictions = np.where(predictions == 1, 0, 1)
            
            try:
                score = silhouette_score(X_scaled, predictions)
                mlflow.log_metric("silhouette_score", score)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except:
                logging.warning("Could not calculate silhouette score")
                continue
                
            n_anomalies = np.sum(predictions == 1)
            mlflow.log_metric("n_anomalies", n_anomalies)
            mlflow.log_metric("anomaly_ratio", n_anomalies / len(predictions))
            
    return best_params

def tune_dbscan(data: pd.DataFrame, param_grid: dict):
    """
    Tune DBSCAN hyperparameters using grid search.
    
    Args:
        data (pd.DataFrame): Input transaction data
        param_grid (dict): Parameter grid to search
        
    Returns:
        dict: Best parameters found
    """
    X = data[FEATURE_COLUMNS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
    
    best_score = -np.inf
    best_params = None
    
    # Grid search
    for params in _get_param_combinations(param_grid):
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.log_params(params)
            
            model = DBSCAN(**params)
            predictions = model.fit_predict(X_scaled)
            predictions = np.where(predictions == -1, 1, 0)  # DBSCAN: -1 is outlier
            
            try:
                score = silhouette_score(X_scaled, predictions)
                mlflow.log_metric("silhouette_score", score)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except:
                logging.warning("Could not calculate silhouette score")
                continue
                
            n_anomalies = np.sum(predictions == 1)
            mlflow.log_metric("n_anomalies", n_anomalies)
            mlflow.log_metric("anomaly_ratio", n_anomalies / len(predictions))
            
    return best_params

def _get_param_combinations(param_grid):
    """Helper function to get all parameter combinations from a grid"""
    keys = param_grid.keys()
    values = param_grid.values()
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance))
