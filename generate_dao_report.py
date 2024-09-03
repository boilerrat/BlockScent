import logging
import os
import pandas as pd
from components.utils import get_db_connection, save_to_csv
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_dao_references():
    """Fetch all records from the database that mention 'DAO'."""
    try:
        conn = get_db_connection()
        if conn is None:
            return pd.DataFrame()
        cur = conn.cursor()

        query = """
        SELECT date, source, headline, sentiment, sentiment_score, label, link
        FROM crypto_news
        WHERE LOWER(headline) LIKE '%dao%';
        """

        cur.execute(query)
        df = pd.DataFrame(cur.fetchall(), columns=['Date', 'Source', 'Headline', 'Sentiment', 'Sentiment Score', 'Label', 'Link'])

        cur.close()
        conn.close()

        return df

    except Exception as e:
        logging.error(f"Error fetching DAO references: {e}")
        return pd.DataFrame()

def generate_dao_report(df, output_dir):
    """Generate and save the DAO report."""
    if not df.empty:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a unique filename based on the current date and time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f"dao_report_{timestamp}.csv")
        
        save_to_csv(df, output_file)
        logging.info(f"DAO report generated and saved to {output_file}.")
    else:
        logging.warning("No DAO references found in the database.")

def main():
    """Main function to fetch, generate, and save the DAO report."""
    df_dao = fetch_dao_references()

    if not df_dao.empty:
        output_dir = '/media/boilerrat/Bobby/CryptoData/BlockScent/csv'
        generate_dao_report(df_dao, output_dir)
    else:
        logging.warning("No DAO references found; report generation skipped.")

if __name__ == "__main__":
    main()
