from jupyter_client import KernelManager
import nbformat as nbf
from pathlib import Path

def create_notebook():
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Add cells
    cells = []
    
    # Title and introduction
    cells.append(nbf.v4.new_markdown_cell('''# Collecte et Comparaison des Données Ethereum

Ce notebook a pour objectif de comparer les données de transactions Ethereum obtenues via deux sources différentes :
1. L'API Etherscan : pour obtenir l'historique des transactions d'une adresse spécifique
2. Web3 : pour obtenir les transactions des blocs récents

Cette comparaison nous permettra de :
- Valider la cohérence des données entre les sources
- Enrichir notre jeu de données avec des informations complémentaires
- Identifier les avantages et limitations de chaque source'''))

    # Imports and setup
    cells.append(nbf.v4.new_code_cell('''import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add src to PYTHONPATH
project_root = Path().absolute().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data_collection import fetch_transactions, fetch_address_transactions_etherscan

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)'''))

    # Section 1: Etherscan
    cells.append(nbf.v4.new_markdown_cell('''## 1. Collecte des données via Etherscan

Nous allons d'abord collecter l'historique des transactions pour une adresse spécifique via l'API Etherscan. 
Pour cet exemple, nous utiliserons l'adresse du hot wallet de Binance : `0x28C6c06298d514Db089934071355E5743bf21d60`'''))

    cells.append(nbf.v4.new_code_cell('''# Adresse du hot wallet de Binance
address = '0x28C6c06298d514Db089934071355E5743bf21d60'

# Collecte des transactions via Etherscan
etherscan_transactions = fetch_address_transactions_etherscan(address)

print(f"Nombre de transactions collectées via Etherscan : {len(etherscan_transactions)}")
etherscan_transactions.head()'''))

    # Section 2: Web3
    cells.append(nbf.v4.new_markdown_cell('''## 2. Collecte des données via Web3

Maintenant, nous allons collecter les transactions des blocs récents via Web3 pour comparer avec les données d'Etherscan.'''))

    cells.append(nbf.v4.new_code_cell('''# Collecte des transactions via Web3
web3_transactions = fetch_transactions()

print(f"Nombre de transactions collectées via Web3 : {len(web3_transactions)}")
web3_transactions.head()'''))

    # Section 3: Comparison
    cells.append(nbf.v4.new_markdown_cell('''## 3. Comparaison et Combinaison des Données

Analysons les différences entre les deux sources de données et créons un jeu de données unifié.'''))

    cells.append(nbf.v4.new_code_cell('''# Comparaison des colonnes
print("Colonnes Etherscan :", etherscan_transactions.columns.tolist())
print("\\nColonnes Web3 :", web3_transactions.columns.tolist())

# Identification des transactions communes
common_txs = pd.merge(
    etherscan_transactions,
    web3_transactions,
    on='hash',
    how='inner',
    suffixes=('_etherscan', '_web3')
)

print(f"\\nNombre de transactions communes : {len(common_txs)}")'''))

    cells.append(nbf.v4.new_code_cell('''# Sauvegarde des données combinées
output_file = RAW_DATA_DIR / f'combined_transactions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
common_txs.to_csv(output_file, index=False)
print(f"Données sauvegardées dans : {output_file}")'''))

    # Add cells to notebook
    nb.cells = cells
    
    # Save the notebook
    nbf.write(nb, 'notebooks/01_data_collection.ipynb')

if __name__ == '__main__':
    create_notebook()
