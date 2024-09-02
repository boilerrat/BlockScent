# fetch_prices.py

import requests
import pandas as pd
import logging
from components.utils import save_to_csv, save_to_database

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_historical_market_data(crypto_id, days):
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart'
    params = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        prices = data['prices']
        market_caps = data['market_caps']
        volumes = data['total_volumes']

        df = pd.DataFrame(prices, columns=['timestamp', f'{crypto_id}_price'])
        df[f'{crypto_id}_market_cap'] = pd.DataFrame(market_caps)[1]
        df[f'{crypto_id}_volume'] = pd.DataFrame(volumes)[1]
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        df.drop('timestamp', axis=1, inplace=True)
        logging.info(f"Successfully fetched {crypto_id} market data.")
        return df
    else:
        logging.error(f"Failed to fetch {crypto_id} market data. Status code: {response.status_code}")
        return pd.DataFrame()

if __name__ == "__main__":
    btc_df = get_historical_market_data('bitcoin', 30)
    eth_df = get_historical_market_data('ethereum', 30)

    if not btc_df.empty and not eth_df.empty:
        merged_df = pd.merge(btc_df, eth_df, on='date', how='outer')
        save_to_database(merged_df, 'crypto_prices')
        save_to_csv(merged_df, '/media/boilerrat/Bobby/CryptoData/BlockScent/csv/crypto_market_data.csv')
    else:
        logging.error("Failed to fetch data for Bitcoin or Ethereum.")
