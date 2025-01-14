import os
from dotenv import load_dotenv

# Chargement des variables d'environnement depuis .env
load_dotenv()

# Configuration des APIs blockchain
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')
INFURA_PROJECT_ID = os.getenv('INFURA_PROJECT_ID')
ALCHEMY_API_KEY = os.getenv('ALCHEMY_API_KEY', '')  # Optional

# URLs des providers
INFURA_URL = f"https://mainnet.infura.io/v3/{INFURA_PROJECT_ID}"
ALCHEMY_URL = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"

# Configuration de la collecte de données
BLOCKS_TO_FETCH = 1000  # Nombre de blocs à analyser
START_BLOCK = 'latest'  # 'latest' ou un numéro de bloc spécifique
BATCH_SIZE = 100  # Nombre de blocs à traiter par lot

# Paramètres de prétraitement
FEATURES = [
    'value',
    'gas_price',
    'gas_used',
    'block_timestamp',
    'transaction_fee'
]

# Paramètres des modèles de détection d'anomalies
ISOLATION_FOREST_PARAMS = {
    'n_estimators': 100,
    'contamination': 0.1,
    'random_state': 42,
    'max_samples': 'auto'
}

DBSCAN_PARAMS = {
    'eps': 0.5,
    'min_samples': 5,
    'metric': 'euclidean'
}

# Configuration du logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Chemins des dossiers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Création des dossiers s'ils n'existent pas
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Vérification de la présence des variables d'environnement requises
def check_environment():
    required_vars = ['ETHERSCAN_API_KEY', 'INFURA_PROJECT_ID']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(
            f"Variables d'environnement manquantes: {', '.join(missing_vars)}. "
            "Veuillez les définir dans le fichier .env"
        )

# Appel de la vérification au chargement du module
check_environment() 