import json

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [source]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source]
    }

# Create notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Add cells
cells = [
    # Title and introduction
    create_markdown_cell("""# Ethereum Transaction Anomaly Detection

This notebook explores anomaly detection on Ethereum transactions using:
- Isolation Forest
- DBSCAN

We'll compare their performance and visualize the results."""),

    # Imports
    create_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Custom imports
from src.preprocessing import transform_data
from src.modeling import anomaly_detection_pipeline, prepare_features, tune_isolation_forest, tune_dbscan

# Plotting settings
plt.style.use('seaborn')
sns.set_palette('Set2')
%matplotlib inline"""),

    # Load and Prepare Data section
    create_markdown_cell("## 1. Load and Prepare Data"),
    
    create_code_cell("""# Load most recent raw data file
raw_data_path = max(Path('../data/raw').glob('transactions_*.csv'))
df_raw = pd.read_csv(raw_data_path)

print(f"Loaded {len(df_raw)} transactions from {raw_data_path.name}")
df_raw.head()"""),

    create_code_cell("""# Transform data and add features
df_processed = transform_data(df_raw)

print("\\nFeatures available:")
for col in df_processed.columns:
    print(f"- {col}")"""),

    # EDA section
    create_markdown_cell("## 2. Exploratory Data Analysis"),

    create_code_cell("""def plot_distribution(df, column, bins=50):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=column, bins=bins)
    plt.title(f'Distribution of {column}')
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=column, bins=bins, log_scale=True)
    plt.title(f'Log Distribution of {column}')
    
    plt.tight_layout()

# Plot key features distributions
for feature in ['value', 'gas_price', 'transaction_fee']:
    plot_distribution(df_processed, feature)

# Plot derived features
for feature in ['log_value', 'value_to_gas_ratio', 'tx_density']:
    plot_distribution(df_processed, feature)"""),

    create_code_cell("""# Correlation heatmap of numerical features
numerical_features = df_processed.select_dtypes(include=[np.number]).columns
correlation = df_processed[numerical_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlations')
plt.tight_layout()"""),

    # Hyperparameter Tuning section
    create_markdown_cell("""## 3. Hyperparameter Tuning

We'll optimize the hyperparameters for both models using cross-validation and our custom silhouette score."""),

    create_code_cell("""# Prepare features for tuning
X_scaled, features_used = prepare_features(df_processed)
print(f"Features prepared for tuning: {features_used}")"""),

    create_markdown_cell("""### 3.1 Isolation Forest Tuning

We'll use GridSearchCV to find the best parameters among:
- n_estimators: [50, 100, 200]
- max_samples: [0.5, 1.0]
- contamination: [0.01, 0.02, 0.05]"""),

    create_code_cell("""# Tune Isolation Forest
print("Starting Isolation Forest tuning...")
if_best_params, if_best_model = tune_isolation_forest(X_scaled)

print("\\nBest Isolation Forest Parameters:")
for param, value in if_best_params.items():
    print(f"- {param}: {value}")"""),

    create_markdown_cell("""### 3.2 DBSCAN Tuning

We'll use RandomizedSearchCV to find the best parameters among:
- eps: [0.3, 0.5, 0.7, 1.0]
- min_samples: [3, 5, 10]"""),

    create_code_cell("""# Tune DBSCAN
print("Starting DBSCAN tuning...")
dbscan_best_params, dbscan_best_model = tune_dbscan(X_scaled)

print("\\nBest DBSCAN Parameters:")
for param, value in dbscan_best_params.items():
    print(f"- {param}: {value}")"""),

    create_markdown_cell("### 3.3 Compare Tuned Models"),

    create_code_cell("""# Get predictions from both tuned models
if_predictions = if_best_model.predict(X_scaled)
if_labels = np.where(if_predictions == 1, 0, 1)

dbscan_predictions = dbscan_best_model.fit_predict(X_scaled)
dbscan_labels = np.where(dbscan_predictions == -1, 1, 0)

# Compare results
results_df = pd.DataFrame({
    'Model': ['Isolation Forest', 'DBSCAN'],
    'Anomalies Detected': [sum(if_labels), sum(dbscan_labels)],
    'Anomaly Percentage': [sum(if_labels)/len(if_labels)*100, sum(dbscan_labels)/len(dbscan_labels)*100]
})

print("Model Comparison after Tuning:")
print(results_df.to_string(index=False))

# Calculate agreement between models
agreement = (if_labels == dbscan_labels).mean() * 100
print(f"\\nModels agree on {agreement:.2f}% of transactions")"""),

    create_code_cell("""# Visualize results from tuned models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot Isolation Forest results
scatter1 = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=if_labels, cmap='coolwarm')
ax1.set_title('Tuned Isolation Forest Anomalies')
ax1.set_xlabel(features_used[0])
ax1.set_ylabel(features_used[1])
plt.colorbar(scatter1, ax=ax1)

# Plot DBSCAN results
scatter2 = ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='coolwarm')
ax2.set_title('Tuned DBSCAN Anomalies')
ax2.set_xlabel(features_used[0])
ax2.set_ylabel(features_used[1])
plt.colorbar(scatter2, ax=ax2)

plt.tight_layout()
plt.show()"""),

    # Anomaly Detection section
    create_markdown_cell("## 4. Anomaly Detection with Default Parameters"),

    create_code_cell("""# Run both models with default parameters
results_if = anomaly_detection_pipeline(df_processed, model_type="IF")
results_dbscan = anomaly_detection_pipeline(df_processed, model_type="DBSCAN")"""),

    create_code_cell("""# Compare results
def compare_anomalies(df1, df2):
    anomalies_if = df1['anomaly_label'].sum()
    anomalies_dbscan = df2['anomaly_label'].sum()
    
    # Calculate agreement between models
    agreement = (df1['anomaly_label'] == df2['anomaly_label']).mean() * 100
    
    print("Model Comparison (Default Parameters):")
    print(f"IsolationForest detected: {anomalies_if} anomalies ({anomalies_if/len(df1)*100:.2f}%)")
    print(f"DBSCAN detected: {anomalies_dbscan} anomalies ({anomalies_dbscan/len(df2)*100:.2f}%)")
    print(f"Models agree on {agreement:.2f}% of transactions")

compare_anomalies(results_if, results_dbscan)"""),

    create_code_cell("""# Visualize anomalies
def plot_anomalies(df, feature_x, feature_y, model_name):
    plt.figure(figsize=(10, 6))
    
    # Plot normal and anomalous points
    normal = df[df['anomaly_label'] == 0]
    anomalies = df[df['anomaly_label'] == 1]
    
    plt.scatter(normal[feature_x], normal[feature_y], 
                c='blue', label='Normal', alpha=0.5, s=50)
    plt.scatter(anomalies[feature_x], anomalies[feature_y], 
                c='red', label='Anomaly', alpha=0.7, s=100)
    
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f'Anomalies detected by {model_name}')
    plt.legend()
    plt.tight_layout()

# Plot for both models
for df, name in [(results_if, 'Isolation Forest'), (results_dbscan, 'DBSCAN')]:
    plot_anomalies(df, 'log_value', 'log_gas_price', name)
    plot_anomalies(df, 'value_to_gas_ratio', 'fee_to_value_ratio', name)"""),

    # Analysis section
    create_markdown_cell("## 5. Analyze Anomalous Transactions"),

    create_code_cell("""# Compare characteristics of normal vs anomalous transactions
def analyze_anomalies(df, model_name):
    normal = df[df['anomaly_label'] == 0]
    anomalies = df[df['anomaly_label'] == 1]
    
    print(f"\\nAnalysis for {model_name}:")
    print("\\nMean values:")
    for col in ['value', 'gas_price', 'transaction_fee']:
        print(f"{col}:")
        print(f"  Normal: {normal[col].mean():.2f}")
        print(f"  Anomalous: {anomalies[col].mean():.2f}")

analyze_anomalies(results_if, "Isolation Forest")
analyze_anomalies(results_dbscan, "DBSCAN")"""),

    # Save Results section
    create_markdown_cell("## 6. Save Results"),

    create_code_cell("""# Combine results from both models
final_df = df_processed.copy()
final_df['anomaly_if'] = results_if['anomaly_label']
final_df['anomaly_dbscan'] = results_dbscan['anomaly_label']
final_df['anomaly_agreement'] = (final_df['anomaly_if'] == final_df['anomaly_dbscan']).astype(int)

# Save to processed directory
output_path = Path('../data/processed/transactions_labeled.csv')
final_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")""")
]

notebook["cells"] = cells

# Save notebook
with open("notebooks/02_modeling.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)