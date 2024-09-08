import logging
import os
import pandas as pd
from components.utils import get_db_connection
from datetime import datetime, timedelta
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_dao_references_last_seven_days():
    """Fetch all records from the database that mention 'DAO' in the last seven days."""
    try:
        conn = get_db_connection()
        if conn is None:
            return pd.DataFrame()
        cur = conn.cursor()

        # Calculate the date range (last seven days)
        today = datetime.now()
        seven_days_ago = today - timedelta(days=7)
        seven_days_ago_str = seven_days_ago.strftime('%Y-%m-%d')

        query = f"""
        SELECT date, source, headline, sentiment, sentiment_score, label, link
        FROM crypto_news
        WHERE LOWER(headline) LIKE '%dao%'
        AND date >= '{seven_days_ago_str}';
        """

        cur.execute(query)
        df = pd.DataFrame(cur.fetchall(), columns=['Date', 'Source', 'Headline', 'Sentiment', 'Sentiment Score', 'Label', 'Link'])

        cur.close()
        conn.close()

        return df

    except Exception as e:
        logging.error(f"Error fetching DAO references: {e}")
        return pd.DataFrame()

def preprocess_text(text):
    """Preprocess the text for word cloud generation."""
    # Remove special characters, numbers, and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def generate_dao_word_cloud(df, output_dir):
    """Generate and save a word cloud image for DAO references."""
    if not df.empty:
        # Preprocess the headlines
        df['processed_headline'] = df['Headline'].apply(preprocess_text)
        all_text = ' '.join(df['processed_headline'].tolist())

        # Generate the word cloud (limit to 15 words)
        wordcloud = WordCloud(width=800, height=400, max_words=15, background_color='white').generate(all_text)

        # Ensure the output directory exists (this check is now redundant since we know it exists)
        os.makedirs(output_dir, exist_ok=True)

        # Generate a unique filename based on the current date and time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_image = os.path.join(output_dir, f"dao_wordcloud_last7days_{timestamp}.png")

        # Save the word cloud image
        wordcloud.to_file(output_image)
        logging.info(f"DAO word cloud image for the last 7 days saved as {output_image}")

        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    else:
        logging.warning("No DAO references found in the last 7 days; word cloud generation skipped.")

def main():
    """Main function to fetch, generate, and save the DAO word cloud for the last 7 days."""
    df_dao = fetch_dao_references_last_seven_days()

    if not df_dao.empty:
        output_dir = '/media/boilerrat/Bobby/CryptoData/BlockScent/charts'
        generate_dao_word_cloud(df_dao, output_dir)
    else:
        logging.warning("No DAO references found in the last 7 days; word cloud generation skipped.")

if __name__ == "__main__":
    main()
