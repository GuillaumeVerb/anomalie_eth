import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import mlflow
from typing import Dict, Any, Tuple

from .config import (
    ISOLATION_FOREST_PARAMS,
    DBSCAN_PARAMS,
    FEATURE_COLUMNS,
    EXPERIMENT_NAME,
    logger
)

def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Preprocess features by scaling them."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[FEATURE_COLUMNS])
    return pd.DataFrame(scaled_features, columns=FEATURE_COLUMNS), scaler

def anomaly_detection_pipeline(
    df: pd.DataFrame,
    model_type: str = "isolation_forest",
    params: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Run anomaly detection pipeline with MLflow tracking.
    
    Args:
        df: Input DataFrame with transaction features
        model_type: Type of model to use ("isolation_forest" or "dbscan")
        params: Model parameters to override defaults
    
    Returns:
        DataFrame with anomaly labels
    """
    # Start MLflow run
    with mlflow.start_run(experiment_name=EXPERIMENT_NAME) as run:
        logger.info(f"Started MLflow run: {run.info.run_id}")
        mlflow.log_param("model_type", model_type)
        
        # Preprocess features
        scaled_df, scaler = preprocess_features(df)
        
        # Select and configure model
        if model_type == "isolation_forest":
            model_params = params or ISOLATION_FOREST_PARAMS
            model = IsolationForest(**model_params)
            mlflow.log_params(model_params)
        elif model_type == "dbscan":
            model_params = params or DBSCAN_PARAMS
            model = DBSCAN(**model_params)
            mlflow.log_params(model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Fit model and predict
        if model_type == "isolation_forest":
            labels = model.fit_predict(scaled_df)
            # Convert to binary labels (1: normal, 0: anomaly)
            labels = np.where(labels == 1, 0, 1)
        else:  # DBSCAN
            labels = model.fit_predict(scaled_df)
            # Convert to binary labels (-1: anomaly, others: normal)
            labels = np.where(labels == -1, 1, 0)
        
        # Calculate silhouette score for non-anomalous points
        try:
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(scaled_df, labels)
                mlflow.log_metric("silhouette_score", silhouette)
                logger.info(f"Silhouette score: {silhouette:.3f}")
        except Exception as e:
            logger.warning(f"Could not calculate silhouette score: {e}")
        
        # Log number of anomalies
        n_anomalies = sum(labels)
        mlflow.log_metric("n_anomalies", n_anomalies)
        mlflow.log_metric("anomaly_ratio", n_anomalies / len(df))
        
        # Add labels to original dataframe
        df['is_anomaly'] = labels
        return df

def tune_isolation_forest(
    df: pd.DataFrame,
    param_grid: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Tune Isolation Forest hyperparameters using GridSearchCV.
    
    Args:
        df: Input DataFrame
        param_grid: Dictionary of parameters to search
        
    Returns:
        Best parameters found
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'contamination': [0.01, 0.05, 0.1],
            'max_samples': ['auto', 100, 500]
        }
    
    scaled_df, _ = preprocess_features(df)
    
    with mlflow.start_run(experiment_name=EXPERIMENT_NAME) as run:
        logger.info(f"Started Isolation Forest tuning run: {run.info.run_id}")
        mlflow.log_params({"tuning_model": "isolation_forest"})
        
        # Create base model
        base_model = IsolationForest(random_state=42)
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1
        )
        
        # Fit GridSearchCV
        grid_search.fit(scaled_df)
        
        # Log results
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_score", grid_search.best_score_)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_params_

def tune_dbscan(
    df: pd.DataFrame,
    param_distributions: Dict[str, Any] = None,
    n_iter: int = 20
) -> Dict[str, Any]:
    """
    Tune DBSCAN hyperparameters using RandomizedSearchCV.
    
    Args:
        df: Input DataFrame
        param_distributions: Dictionary of parameters to sample from
        n_iter: Number of parameter settings to try
        
    Returns:
        Best parameters found
    """
    if param_distributions is None:
        param_distributions = {
            'eps': np.linspace(0.1, 2.0, 20),
            'min_samples': range(2, 11)
        }
    
    scaled_df, _ = preprocess_features(df)
    
    with mlflow.start_run(experiment_name=EXPERIMENT_NAME) as run:
        logger.info(f"Started DBSCAN tuning run: {run.info.run_id}")
        mlflow.log_params({"tuning_model": "dbscan"})
        
        # Create base model
        base_model = DBSCAN()
        
        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1
        )
        
        # Fit RandomizedSearchCV
        random_search.fit(scaled_df)
        
        # Log results
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("best_score", random_search.best_score_)
        
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best score: {random_search.best_score_:.3f}")
        
        return random_search.best_params_
