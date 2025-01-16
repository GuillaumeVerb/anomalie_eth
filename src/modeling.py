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

def prepare_features(data: pd.DataFrame):
    """
    Prepare and scale features for anomaly detection.
    
    Args:
        data (pd.DataFrame): Input transaction data
        
    Returns:
        tuple: (scaled_features, feature_names)
            - scaled_features: numpy array of scaled features
            - feature_names: list of feature names used
    """
    # Extract features
    X = data[FEATURE_COLUMNS].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, list(FEATURE_COLUMNS)

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
    
    # Initialize predictions as None
    predictions = None
    run_id = None
    
    try:
        # End any active runs to avoid nested run errors
        active_run = mlflow.active_run()
        if active_run:
            run_id = active_run.info.run_id
            mlflow.end_run()
        
        # Get experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            logging.error(f"Experiment {EXPERIMENT_NAME} not found")
            raise ValueError(f"Experiment {EXPERIMENT_NAME} not found")
            
        # Start MLflow run
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="prediction", tags={
            "run_type": "prediction",
            "model_type": model_type,
            "parent_run_id": run_id if run_id else "none",
            "status": "running"
        }) as run:
            try:
                # Log dataset info
                mlflow.log_param("n_samples", len(data))
                mlflow.log_param("features", FEATURE_COLUMNS)
                mlflow.log_param("model_type", model_type)
                
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
                    
                # Calculate silhouette score if we have both classes
                try:
                    unique_labels = np.unique(predictions)
                    mlflow.log_param("unique_labels", list(unique_labels))
                    if len(unique_labels) > 1:
                        silhouette = silhouette_score(X_scaled, predictions)
                        mlflow.log_metric("silhouette_score", silhouette)
                        logging.info(f"Silhouette score: {silhouette:.3f}")
                except Exception as e:
                    logging.warning(f"Could not calculate silhouette score: {str(e)}")
                    mlflow.set_tag("silhouette_error", str(e))
                    
                # Log number of anomalies
                n_anomalies = np.sum(predictions)
                mlflow.log_metric("n_anomalies", n_anomalies)
                mlflow.log_metric("anomaly_ratio", n_anomalies / len(predictions))
                
                logging.info(f"Detected {n_anomalies} anomalies ({n_anomalies/len(predictions)*100:.1f}%)")
                
                # Mark run as successful
                mlflow.set_tag("status", "completed")
                
            except Exception as e:
                # Mark run as failed and log error
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(e))
                raise
            
    except Exception as e:
        logging.warning(f"MLflow tracking failed: {str(e)}")
        # Continue without MLflow tracking if predictions haven't been made yet
        if predictions is None:
            if model_type.upper() in ["IF", "ISOLATION_FOREST"]:
                model = IsolationForest(**ISOLATION_FOREST_PARAMS)
            else:
                model = DBSCAN(**DBSCAN_PARAMS)
                
            predictions = model.fit_predict(X_scaled)
            predictions = predictions == -1
    finally:
        # Ensure any active run is ended
        try:
            active_run = mlflow.active_run()
            if active_run and active_run.info.run_id != run_id:
                mlflow.end_run()
        except Exception as e:
            logging.warning(f"Error ending MLflow run: {str(e)}")
    
    # Add predictions to original data
    result_df = data.copy()
    result_df['anomaly_label'] = predictions.astype(int)  # Convert boolean to int (0: normal, 1: anomaly)
        
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
    
    try:
        # End any active runs to avoid nested run errors
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()
        
        # Get experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            logging.error(f"Experiment {EXPERIMENT_NAME} not found")
            raise ValueError(f"Experiment {EXPERIMENT_NAME} not found")
        
        best_score = -np.inf
        best_params = None
        
        # Grid search
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="IF_tuning", tags={"run_type": "tuning", "model": "IsolationForest"}) as parent_run:
            mlflow.log_param("model_type", "IsolationForest")
            mlflow.log_param("tuning_params", list(param_grid.keys()))
            mlflow.log_param("n_combinations", len(list(_get_param_combinations(param_grid))))
            
            for i, params in enumerate(_get_param_combinations(param_grid)):
                run_name = f"IF_trial_{i+1}"
                with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name, nested=True, tags={"run_type": "trial", "model": "IsolationForest", "trial_number": str(i+1)}) as child_run:
                    mlflow.log_params(params)
                    
                    model = IsolationForest(**params)
                    predictions = model.fit_predict(X_scaled)
                    predictions = predictions == -1  # Convert to boolean (True: anomaly, False: normal)
                    
                    try:
                        unique_labels = np.unique(predictions)
                        mlflow.log_param("unique_labels", list(unique_labels))
                        if len(unique_labels) > 1:
                            score = silhouette_score(X_scaled, predictions)
                            mlflow.log_metric("silhouette_score", score)
                            
                            if score > best_score:
                                best_score = score
                                best_params = params.copy()
                                mlflow.log_metric("best_score_so_far", score)
                                mlflow.set_tag("is_best", "true")
                            else:
                                mlflow.set_tag("is_best", "false")
                    except Exception as e:
                        logging.warning(f"Could not calculate silhouette score for trial {i+1}: {str(e)}")
                        mlflow.set_tag("error", str(e))
                        continue
                        
                    n_anomalies = np.sum(predictions)
                    mlflow.log_metric("n_anomalies", n_anomalies)
                    mlflow.log_metric("anomaly_ratio", n_anomalies / len(predictions))
                    
            # Log best parameters in parent run
            if best_params:
                mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
                mlflow.log_metric("final_best_score", best_score)
                
    except Exception as e:
        logging.warning(f"MLflow tracking failed in tuning: {str(e)}")
        # Return default parameters if MLflow fails
        best_params = ISOLATION_FOREST_PARAMS
    finally:
        try:
            active_run = mlflow.active_run()
            if active_run:
                mlflow.end_run()
        except Exception as e:
            logging.warning(f"Error ending MLflow run: {str(e)}")
            
    return best_params or ISOLATION_FOREST_PARAMS

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
    
    try:
        # End any active runs to avoid nested run errors
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()
        
        # Get experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            logging.error(f"Experiment {EXPERIMENT_NAME} not found")
            raise ValueError(f"Experiment {EXPERIMENT_NAME} not found")
        
        best_score = -np.inf
        best_params = None
        
        # Grid search
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="DBSCAN_tuning", tags={"run_type": "tuning", "model": "DBSCAN"}) as parent_run:
            mlflow.log_param("model_type", "DBSCAN")
            mlflow.log_param("tuning_params", list(param_grid.keys()))
            mlflow.log_param("n_combinations", len(list(_get_param_combinations(param_grid))))
            
            for i, params in enumerate(_get_param_combinations(param_grid)):
                run_name = f"DBSCAN_trial_{i+1}"
                with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name, nested=True, tags={"run_type": "trial", "model": "DBSCAN", "trial_number": str(i+1)}) as child_run:
                    mlflow.log_params(params)
                    
                    model = DBSCAN(**params)
                    predictions = model.fit_predict(X_scaled)
                    predictions = predictions == -1  # Convert to boolean (True: anomaly, False: normal)
                    
                    try:
                        unique_labels = np.unique(predictions)
                        mlflow.log_param("unique_labels", list(unique_labels))
                        if len(unique_labels) > 1:
                            score = silhouette_score(X_scaled, predictions)
                            mlflow.log_metric("silhouette_score", score)
                            
                            if score > best_score:
                                best_score = score
                                best_params = params.copy()
                                mlflow.log_metric("best_score_so_far", score)
                                mlflow.set_tag("is_best", "true")
                            else:
                                mlflow.set_tag("is_best", "false")
                    except Exception as e:
                        logging.warning(f"Could not calculate silhouette score for trial {i+1}: {str(e)}")
                        mlflow.set_tag("error", str(e))
                        continue
                        
                    n_anomalies = np.sum(predictions)
                    mlflow.log_metric("n_anomalies", n_anomalies)
                    mlflow.log_metric("anomaly_ratio", n_anomalies / len(predictions))
                    
            # Log best parameters in parent run
            if best_params:
                mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
                mlflow.log_metric("final_best_score", best_score)
                
    except Exception as e:
        logging.warning(f"MLflow tracking failed in tuning: {str(e)}")
        # Return default parameters if MLflow fails
        best_params = DBSCAN_PARAMS
    finally:
        try:
            active_run = mlflow.active_run()
            if active_run:
                mlflow.end_run()
        except Exception as e:
            logging.warning(f"Error ending MLflow run: {str(e)}")
            
    return best_params or DBSCAN_PARAMS

def _get_param_combinations(param_grid):
    """Helper function to get all parameter combinations from a grid"""
    keys = param_grid.keys()
    values = param_grid.values()
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance))
