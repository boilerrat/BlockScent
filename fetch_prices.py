import os
import psycopg2
import requests
import pandas as pd
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def get_historical_market_data(crypto_id, days):
    """
    Fetches historical prices, market caps, and trading volumes for a given cryptocurrency from CoinGecko API.
    """
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart'
    params = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        prices = data['prices']
        market_caps = data['market_caps']
        volumes = data['total_volumes']

        # Convert to DataFrame
        df = pd.DataFrame(prices, columns=['timestamp', f'{crypto_id}_price'])
        df[f'{crypto_id}_market_cap'] = pd.DataFrame(market_caps)[1]
        df[f'{crypto_id}_volume'] = pd.DataFrame(volumes)[1]
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        df.drop('timestamp', axis=1, inplace=True)
        logging.info(f"Successfully fetched {crypto_id} market data.")
        return df
    else:
        logging.error(f"Failed to fetch {crypto_id} market data. Status code: {response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame if the API call fails

def save_to_database(df, table_name):
    """
    Saves a DataFrame to the specified PostgreSQL table.
    """
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()

        # Create the table if it doesn't exist
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            date DATE PRIMARY KEY,
            btc_price NUMERIC,
            btc_market_cap NUMERIC,
            btc_volume NUMERIC,
            eth_price NUMERIC,
            eth_market_cap NUMERIC,
            eth_volume NUMERIC
        );
        """
        cur.execute(create_table_query)
        conn.commit()

        # Insert data into the table
        for _, row in df.iterrows():
            insert_query = f"""
            INSERT INTO {table_name} (date, btc_price, btc_market_cap, btc_volume, eth_price, eth_market_cap, eth_volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date) DO NOTHING;
            """
            cur.execute(insert_query, (
                row['date'], 
                row.get('bitcoin_price', None), 
                row.get('bitcoin_market_cap', None), 
                row.get('bitcoin_volume', None),
                row.get('ethereum_price', None), 
                row.get('ethereum_market_cap', None), 
                row.get('ethereum_volume', None)
            ))
        conn.commit()

        logging.info(f"Data saved to the {table_name} table.")
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Error saving to database: {e}")

def save_to_csv(df, filename):
    """
    Saves the DataFrame to a CSV file.
    """
    try:
        df.to_csv(filename, index=False)
        logging.info(f"Data saved to {filename}.")
    except Exception as e:
        logging.error(f"Error saving to CSV: {e}")

if __name__ == "__main__":
    # Fetch historical market data for Bitcoin (btc) and Ethereum (eth) for the past 30 days
    btc_df = get_historical_market_data('bitcoin', 30)
    eth_df = get_historical_market_data('ethereum', 30)

    if not btc_df.empty and not eth_df.empty:
        # Merge the two DataFrames on the 'date' column
        merged_df = pd.merge(btc_df, eth_df, on='date', how='outer')

        # Save to PostgreSQL database
        save_to_database(merged_df, 'crypto_market_data')

        # Save to CSV
        save_to_csv(merged_df, '/media/boilerrat/Bobby/CryptoData/BlockScent/csv/crypto_market_data.csv')
    else:
        logging.error("Failed to fetch data for Bitcoin or Ethereum.")
