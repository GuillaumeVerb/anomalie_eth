import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import logging
import sys
from pathlib import Path

from src.config import (
    ISOLATION_FOREST_PARAMS,
    DBSCAN_PARAMS,
    MODEL_DIR,
    LOG_LEVEL,
    LOG_FORMAT,
    FEATURES
)

# Logging configuration
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def prepare_features(df):
    """
    Select and prepare features for anomaly detection.
    
    Args:
        df (pd.DataFrame): Input DataFrame with all features
        
    Returns:
        np.array: Scaled feature matrix
    """
    try:
        # Select relevant features
        feature_cols = [
            'log_value', 
            'log_gas_price', 
            'log_gas_used',
            'log_transaction_fee',
            'from_address_freq',
            'to_address_freq',
            'value_to_gas_ratio',
            'fee_to_value_ratio',
            'tx_density'
        ]
        
        # Ensure all features exist
        available_features = [col for col in feature_cols if col in df.columns]
        if not available_features:
            raise ValueError("No required features found in DataFrame")
        
        # Extract features
        X = df[available_features].copy()
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logger.info(f"Prepared {len(available_features)} features for modeling")
        return X_scaled, available_features
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise

def train_isolation_forest(X):
    """Train IsolationForest model with configured parameters."""
    try:
        model = IsolationForest(**ISOLATION_FOREST_PARAMS)
        predictions = model.fit_predict(X)
        # Convert to binary labels (1: normal, 0: anomaly)
        labels = np.where(predictions == 1, 0, 1)
        return labels, model
        
    except Exception as e:
        logger.error(f"Error training IsolationForest: {str(e)}")
        raise

def train_dbscan(X):
    """Train DBSCAN model with configured parameters."""
    try:
        model = DBSCAN(**DBSCAN_PARAMS)
        labels = model.fit_predict(X)
        # Convert to binary labels (0: normal, 1: anomaly)
        labels = np.where(labels == -1, 1, 0)
        return labels, model
        
    except Exception as e:
        logger.error(f"Error training DBSCAN: {str(e)}")
        raise

def calculate_silhouette(X, labels):
    """
    Calculate silhouette score, excluding anomalies.
    """
    try:
        # Filter out anomalies (label 1)
        mask = labels == 0
        if sum(mask) < 2:
            logger.warning("Not enough normal points to calculate silhouette score")
            return None
            
        # Calculate score only for normal points
        score = silhouette_score(X[mask], labels[mask])
        return score
        
    except Exception as e:
        logger.warning(f"Could not calculate silhouette score: {str(e)}")
        return None

def anomaly_detection_pipeline(df, model_type="IF"):
    """
    Complete anomaly detection pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame with features
        model_type (str): Type of model to use ("IF" for IsolationForest or "DBSCAN")
    
    Returns:
        pd.DataFrame: Original DataFrame with anomaly labels added
    """
    try:
        logger.info(f"Starting anomaly detection with {model_type}")
        
        # Prepare features
        X_scaled, features_used = prepare_features(df)
        
        # Train model and get predictions
        if model_type.upper() == "IF":
            labels, model = train_isolation_forest(X_scaled)
        elif model_type.upper() == "DBSCAN":
            labels, model = train_dbscan(X_scaled)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Add predictions to DataFrame
        df_result = df.copy()
        df_result['anomaly_label'] = labels
        
        # Calculate metrics
        n_anomalies = sum(labels)
        anomaly_percentage = (n_anomalies / len(labels)) * 100
        silhouette = calculate_silhouette(X_scaled, labels)
        
        # Log results
        logger.info("Anomaly detection completed:")
        logger.info(f"- Features used: {features_used}")
        logger.info(f"- Anomalies detected: {n_anomalies} ({anomaly_percentage:.2f}%)")
        if silhouette is not None:
            logger.info(f"- Silhouette score: {silhouette:.3f}")
        
        # Save anomalous transactions to a separate file
        anomalies_df = df_result[df_result['anomaly_label'] == 1]
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(MODEL_DIR) / f"anomalies_{model_type}_{timestamp}.csv"
        anomalies_df.to_csv(output_file, index=False)
        logger.info(f"Anomalous transactions saved to {output_file}")
        
        return df_result
        
    except Exception as e:
        logger.error(f"Error in anomaly detection pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Example usage
        from src.preprocessing import transform_data
        
        # Load and preprocess data
        input_file = max(Path("data/raw").glob("transactions_*.csv"))
        df = pd.read_csv(input_file)
        df_processed = transform_data(df)
        
        # Run anomaly detection with both models
        results_if = anomaly_detection_pipeline(df_processed, model_type="IF")
        results_dbscan = anomaly_detection_pipeline(df_processed, model_type="DBSCAN")
        
        # Compare results
        if_anomalies = results_if['anomaly_label'].sum()
        dbscan_anomalies = results_dbscan['anomaly_label'].sum()
        
        logger.info("\nComparison of models:")
        logger.info(f"IsolationForest detected: {if_anomalies} anomalies")
        logger.info(f"DBSCAN detected: {dbscan_anomalies} anomalies")
        
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        sys.exit(1)
