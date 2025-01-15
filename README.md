# Ethereum Transaction Anomaly Detection

## Etherscan API Key

Pour enrichir les données de transactions avec des informations supplémentaires, nous utilisons l'API Etherscan. Voici comment configurer votre clé API :

1. Créez un compte sur [etherscan.io](https://etherscan.io/register)
2. Une fois connecté, allez dans votre profil et sélectionnez "API-KEYs"
3. Cliquez sur "Add" pour générer une nouvelle clé API
4. Copiez la clé générée et ajoutez-la dans votre fichier `.env` :
   ```
   ETHERSCAN_API_KEY="votre-clé-api"
   ```

⚠️ **Important** : Ne committez jamais votre fichier `.env` contenant votre clé API. Il est déjà inclus dans `.gitignore`.

### Utilisation optionnelle de l'API Etherscan

Avec votre clé API, vous pouvez enrichir vos données de transactions avec des informations supplémentaires comme :
- Les labels des adresses (exchanges, smart contracts connus, etc.)
- Les ABI des smart contracts pour décoder les transactions
- Les balances historiques des adresses
- Les tags de transactions (DEX trades, NFT transfers, etc.)

Exemple d'utilisation de l'API pour obtenir des informations sur une adresse :
```python
from etherscan import Etherscan
eth = Etherscan(api_key)

# Obtenir le label d'une adresse
address_info = eth.get_contract_source_code('0x...')

# Obtenir l'ABI d'un contrat
contract_abi = eth.get_contract_abi('0x...')

# Obtenir l'historique des transactions
txs = eth.get_normal_txs_by_address('0x...', startblock=0, endblock=99999999)
```
