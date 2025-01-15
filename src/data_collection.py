import pandas as pd
from web3 import Web3
import logging
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime

from src.config import (
    ETH_NODE_URL,
    RAW_DATA_DIR,
    logger
)

def fetch_transactions(
    start_block='latest',
    num_blocks=100,
    batch_size=10,
    output_file=None
):
    """
    Fetch Ethereum transactions and save them to a CSV file.
    
    Args:
        start_block (str or int): Starting block number or 'latest'
        num_blocks (int): Number of blocks to fetch
        batch_size (int): Number of blocks to process in each batch
        output_file (str): Path to save the CSV file
    """
    # Connect to Ethereum node
    w3 = Web3(Web3.HTTPProvider(ETH_NODE_URL))
    if not w3.is_connected():
        raise ConnectionError("Failed to connect to Ethereum node")
    
    # Get the latest block number if start_block is 'latest'
    if start_block == 'latest':
        start_block = w3.eth.block_number
    
    # Calculate block range
    end_block = start_block - num_blocks
    current_block = start_block
    
    # Prepare output file
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = RAW_DATA_DIR / f"transactions_{timestamp}.csv"
    
    transactions = []
    
    try:
        with tqdm(total=num_blocks, desc="Fetching blocks") as pbar:
            while current_block > end_block:
                batch_end = max(current_block - batch_size, end_block)
                
                # Process each block in the batch
                for block_num in range(current_block, batch_end, -1):
                    try:
                        # Get block data
                        block = w3.eth.get_block(block_num, full_transactions=True)
                        
                        # Process transactions in the block
                        for tx in block.transactions:
                            tx_dict = {
                                'hash': tx['hash'].hex(),
                                'block_number': block_num,
                                'from_address': tx['from'],
                                'to_address': tx.get('to', None),
                                'value': float(w3.from_wei(tx['value'], 'ether')),
                                'gas': tx['gas'],
                                'block_timestamp': block.timestamp
                            }
                            
                            # Handle gas price (might be maxFeePerGas in EIP-1559 transactions)
                            if 'gasPrice' in tx:
                                gas_price = float(w3.from_wei(tx['gasPrice'], 'gwei'))
                            elif 'maxFeePerGas' in tx:
                                gas_price = float(w3.from_wei(tx['maxFeePerGas'], 'gwei'))
                            else:
                                gas_price = None
                            
                            tx_dict['gas_price'] = gas_price
                            
                            # Calculate transaction fee if gas price is available
                            if gas_price is not None:
                                tx_dict['transaction_fee'] = gas_price * tx['gas']
                            else:
                                tx_dict['transaction_fee'] = None
                            
                            transactions.append(tx_dict)
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing block {block_num}: {str(e)}")
                        continue
                
                current_block = batch_end
                
                # Save batch to CSV
                if len(transactions) > 0:
                    df = pd.DataFrame(transactions)
                    df.to_csv(output_file, index=False, mode='a', header=not output_file.exists())
                    transactions = []  # Clear the list after saving
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
        
        logger.info(f"Data collection completed. Transactions saved to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error in data collection: {str(e)}")
        raise

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Fetch transactions
    fetch_transactions(
        start_block='latest',
        num_blocks=100,
        batch_size=10
    )
