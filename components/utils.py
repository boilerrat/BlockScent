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
        os.makedirs(os.path.dirname(filename), exist_ok=True)
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

        # Ensure table exists with proper schema
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            date DATE,
            source VARCHAR(255),
            headline TEXT,
            sentiment VARCHAR(50),
            sentiment_score NUMERIC,
            label VARCHAR(50),
            link TEXT,
            UNIQUE (date, source, headline, link)
        );
        """)
        conn.commit()

        # Insert data, avoid duplication
        for _, row in df.iterrows():
            cur.execute(f"""
            INSERT INTO {table_name} (date, source, headline, sentiment, sentiment_score, label, link)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, source, headline, link) DO NOTHING;
            """, tuple(row))
        conn.commit()

        logging.info(f"Data saved to the {table_name} table.")
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Error saving to database: {e}")

def filter_headlines_by_keyword(df, keyword, output_filename):
    """Filter headlines by keyword and save to CSV."""
    filtered_df = df[df['Headline'].str.contains(keyword, case=False, na=False)]
    if not filtered_df.empty:
        save_to_csv(filtered_df, output_filename)
        logging.info(f"Filtered data containing keyword '{keyword}' saved to {output_filename}")
    else:
        logging.warning(f"No headlines found containing the keyword '{keyword}'")
