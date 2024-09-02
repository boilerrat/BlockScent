# utils.py

import logging
import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_connection():
    """Establish and return a database connection."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        return None

def save_to_csv(df, filename):
    """Save DataFrame to CSV."""
    try:
        df.to_csv(filename, index=False)
        logging.info(f"Data saved to {filename}.")
    except Exception as e:
        logging.error(f"Error saving to CSV: {e}")

def save_to_database(df, table_name):
    """Save DataFrame to PostgreSQL database."""
    try:
        conn = get_db_connection()
        if conn is None:
            return

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
                row.get('btc_price', None), 
                row.get('btc_market_cap', None), 
                row.get('btc_volume', None),
                row.get('eth_price', None), 
                row.get('eth_market_cap', None), 
                row.get('eth_volume', None)
            ))
        conn.commit()

        logging.info(f"Data saved to the {table_name} table.")
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Error saving to database: {e}")

# Use these functions in your main scripts:
# from utils import save_to_csv, save_to_database
