import os
import requests
import time
from dotenv import load_dotenv
import logging
from typing import Dict, List, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')

if not ETHERSCAN_API_KEY:
    raise ValueError("ETHERSCAN_API_KEY not found in .env file")

# Rate limiting configuration (5 calls per second for free API)
RATE_LIMIT = 0.2  # seconds between calls

class EtherscanAPIError(Exception):
    """Custom exception for Etherscan API errors"""
    pass

_last_call_time = 0

def _rate_limit():
    """Implement rate limiting for API calls"""
    global _last_call_time
    current_time = time.time()
    time_since_last_call = current_time - _last_call_time
    
    if time_since_last_call < RATE_LIMIT:
        time.sleep(RATE_LIMIT - time_since_last_call)
    
    _last_call_time = time.time()

def _get_api_url(network: str = 'mainnet') -> str:
    """Get the appropriate Etherscan API URL based on the network."""
    if network == 'mainnet':
        return 'https://api.etherscan.io/api'
    elif network == 'goerli':
        return 'https://api-goerli.etherscan.io/api'
    elif network == 'sepolia':
        return 'https://api-sepolia.etherscan.io/api'
    else:
        raise ValueError(f"Unsupported network: {network}")

def _make_request(params: Dict, network: str = 'mainnet') -> Dict:
    """Make a request to the Etherscan API with error handling and rate limiting."""
    _rate_limit()  # Apply rate limiting
    url = _get_api_url(network)
    params['apikey'] = ETHERSCAN_API_KEY
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        if data['status'] == '0':
            raise EtherscanAPIError(f"API Error: {data.get('message', 'Unknown error')}")
            
        return data['result']
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise EtherscanAPIError(f"Request failed: {str(e)}")

def get_address_balance(address: str, network: str = 'mainnet') -> float:
    """Get the current balance of an Ethereum address in Ether."""
    params = {
        'module': 'account',
        'action': 'balance',
        'address': address,
        'tag': 'latest'
    }
    
    try:
        result = _make_request(params, network)
        balance_eth = float(result) / 10**18
        logger.info(f"Retrieved balance for address {address}: {balance_eth} ETH")
        return balance_eth
        
    except Exception as e:
        logger.error(f"Failed to get balance for address {address}: {str(e)}")
        raise

def get_address_transactions(
    address: str,
    start_block: int = 0,
    end_block: int = 99999999,
    network: str = 'mainnet'
) -> List[Dict]:
    """Get the transaction history of an Ethereum address."""
    params = {
        'module': 'account',
        'action': 'txlist',
        'address': address,
        'startblock': start_block,
        'endblock': end_block,
        'sort': 'desc'
    }
    
    try:
        transactions = _make_request(params, network)
        logger.info(f"Retrieved {len(transactions)} transactions for address {address}")
        return transactions
        
    except Exception as e:
        logger.error(f"Failed to get transactions for address {address}: {str(e)}")
        raise

def get_contract_abi(contract_address: str, network: str = 'mainnet') -> Dict:
    """
    Get the ABI for a verified smart contract.
    
    Args:
        contract_address (str): The contract address
        network (str): The network to use
    
    Returns:
        Dict: Contract ABI
        
    Raises:
        EtherscanAPIError: If the contract is not verified or API request fails
    """
    params = {
        'module': 'contract',
        'action': 'getabi',
        'address': contract_address
    }
    
    try:
        abi = _make_request(params, network)
        logger.info(f"Retrieved ABI for contract {contract_address}")
        return abi
        
    except Exception as e:
        logger.error(f"Failed to get ABI for contract {contract_address}: {str(e)}")
        raise

def get_address_label(address: str, network: str = 'mainnet') -> Optional[str]:
    """
    Get the label for a known Ethereum address (if available).
    
    Args:
        address (str): The Ethereum address to query
        network (str): The network to use
    
    Returns:
        Optional[str]: The address label if available, None otherwise
    """
    params = {
        'module': 'contract',
        'action': 'getsourcecode',
        'address': address
    }
    
    try:
        result = _make_request(params, network)
        if result and isinstance(result, list) and len(result) > 0:
            contract_name = result[0].get('ContractName')
            if contract_name and contract_name != '':
                logger.info(f"Retrieved label for address {address}: {contract_name}")
                return contract_name
        return None
        
    except Exception as e:
        logger.error(f"Failed to get label for address {address}: {str(e)}")
        return None

def get_transaction_receipt(tx_hash: str, network: str = 'mainnet') -> Dict:
    """
    Get the receipt for a specific transaction.
    
    Args:
        tx_hash (str): The transaction hash
        network (str): The network to use
    
    Returns:
        Dict: Transaction receipt details
        
    Raises:
        EtherscanAPIError: If the API request fails
    """
    params = {
        'module': 'proxy',
        'action': 'eth_getTransactionReceipt',
        'txhash': tx_hash
    }
    
    try:
        receipt = _make_request(params, network)
        logger.info(f"Retrieved receipt for transaction {tx_hash}")
        return receipt
        
    except Exception as e:
        logger.error(f"Failed to get receipt for transaction {tx_hash}: {str(e)}")
        raise

def get_gas_oracle(network: str = 'mainnet') -> Dict:
    """
    Get current gas prices from Etherscan gas tracker.
    
    Args:
        network (str): The network to use
    
    Returns:
        Dict: Current gas prices (SafeLow, Standard, Fast)
        
    Raises:
        EtherscanAPIError: If the API request fails
    """
    params = {
        'module': 'gastracker',
        'action': 'gasoracle'
    }
    
    try:
        gas_prices = _make_request(params, network)
        logger.info("Retrieved current gas prices from oracle")
        return gas_prices
        
    except Exception as e:
        logger.error(f"Failed to get gas prices: {str(e)}")
        raise 