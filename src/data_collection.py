import pandas as pd
from web3 import Web3
import time
from pathlib import Path
import logging
from tqdm import tqdm
import sys
from datetime import datetime

from src.config import (
    INFURA_URL,
    BLOCKS_TO_FETCH,
    BATCH_SIZE,
    START_BLOCK,
    RAW_DATA_DIR,
    LOG_LEVEL,
    LOG_FORMAT
)

# Logging configuration
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def connect_to_ethereum():
    """Establishes a connection with an Ethereum node."""
    try:
        w3 = Web3(Web3.HTTPProvider(INFURA_URL))
        if not w3.is_connected():
            raise ConnectionError("Unable to connect to Ethereum node")
        logger.info("Successfully connected to Ethereum node")
        return w3
    except Exception as e:
        logger.error(f"Connection error to Ethereum node: {str(e)}")
        raise

def get_block_transactions(w3, block_number):
    """Retrieves all transactions from a given block."""
    try:
        block = w3.eth.get_block(block_number, full_transactions=True)
        transactions = []
        
        for tx in block.transactions:
            transaction = {
                'hash': tx['hash'].hex(),
                'block_number': block_number,
                'from_address': tx['from'],
                'to_address': tx.get('to', None),  # Can be None for contract creations
                'value': float(w3.from_wei(tx['value'], 'ether')),
                'gas_price': float(w3.from_wei(tx['gasPrice'], 'gwei')),
                'gas_used': tx['gas'],
                'block_timestamp': datetime.fromtimestamp(block.timestamp).isoformat(),
                'transaction_fee': float(w3.from_wei(tx['gasPrice'] * tx['gas'], 'ether')),
                'nonce': tx['nonce'],
                'input_data': tx['input'] if tx['input'] != '0x' else None
            }
            transactions.append(transaction)
            
        return transactions
    except Exception as e:
        logger.error(f"Error retrieving block {block_number}: {str(e)}")
        return []

def fetch_transactions():
    """
    Fetches transactions from the N latest blocks and saves them to a CSV file.
    Returns:
        Path: Path to the CSV file containing the transactions.
    Raises:
        ConnectionError: If connection to Ethereum node fails.
        Exception: For any other error during collection.
    """
    try:
        # Connect to Ethereum node
        w3 = connect_to_ethereum()

        # Determine starting block
        latest_block = w3.eth.block_number
        start_block = latest_block - BLOCKS_TO_FETCH if START_BLOCK == 'latest' else START_BLOCK
        
        logger.info(f"Starting collection from block {start_block} to {latest_block}")
        
        all_transactions = []
        failed_blocks = []
        
        # Use tqdm for progress bar
        for block_number in tqdm(range(start_block, latest_block + 1), 
                               desc="Collecting blocks",
                               total=BLOCKS_TO_FETCH):
            
            transactions = get_block_transactions(w3, block_number)
            
            if transactions:
                all_transactions.extend(transactions)
            else:
                failed_blocks.append(block_number)
            
            # Short pause to avoid API rate limiting
            time.sleep(0.1)
            
            # Intermediate logging every BATCH_SIZE blocks
            if (block_number - start_block + 1) % BATCH_SIZE == 0:
                logger.info(f"Progress: {block_number - start_block + 1}/{BLOCKS_TO_FETCH} blocks processed")

        # Create DataFrame
        df = pd.DataFrame(all_transactions)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(RAW_DATA_DIR) / f"transactions_{timestamp}.csv"
        
        # Create directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save data
        df.to_csv(output_file, index=False)
        
        # Final logs
        total_transactions = len(df)
        avg_tx_per_block = total_transactions / (BLOCKS_TO_FETCH - len(failed_blocks))
        
        logger.info("Collection completed:")
        logger.info(f"- {total_transactions} transactions saved to {output_file}")
        logger.info(f"- Average of {avg_tx_per_block:.2f} transactions per block")
        logger.info(f"- {len(failed_blocks)} failed blocks out of {BLOCKS_TO_FETCH}")
        
        if failed_blocks:
            logger.warning(f"Failed blocks: {failed_blocks}")
        
        return output_file

    except Exception as e:
        logger.error(f"Error during transaction collection: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        output_file = fetch_transactions()
        print(f"\nTransactions have been saved to: {output_file}")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        sys.exit(1)
