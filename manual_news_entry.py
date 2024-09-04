import logging
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from components.utils import get_db_connection, save_to_csv
from datetime import datetime
import re
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load BERT model and tokenizer once
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Paths to CSV files
MAIN_CSV_PATH = '/media/boilerrat/Bobby/CryptoData/BlockScent/csv/crypto_news_sentiment.csv'
DAO_CSV_PATH = '/media/boilerrat/Bobby/CryptoData/BlockScent/csv/crypto_news_sentiment_DAO.csv'

def analyze_sentiment(content):
    """Analyzes sentiment using BERT."""
    inputs = tokenizer(content, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    sentiment_scores = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
    sentiment_label = sentiment_scores.argmax() + 1
    sentiment_score = round(float(sentiment_scores[sentiment_label - 1]), 4)
    sentiment = 'Positive' if sentiment_score > 0.5 else 'Negative'
    return sentiment, sentiment_score, f"{sentiment_label} stars"

def get_manual_entry():
    """Collects manual entry details from the user."""
    date_input = input("Enter the publication date (YYYY-MM-DD): ")
    while not re.match(r"\d{4}-\d{2}-\d{2}", date_input):
        print("Invalid date format. Please enter in YYYY-MM-DD format.")
        date_input = input("Enter the publication date (YYYY-MM-DD): ")
    date = datetime.strptime(date_input, "%Y-%m-%d").date()

    source = input("Enter the news source (e.g., CoinDesk): ")
    headline = input("Enter the headline: ")
    content = input("Enter the content of the news story: ")
    link = input("Enter the URL link to the story: ")

    sentiment, sentiment_score, label = analyze_sentiment(content)

    entry = {
        'Date': date,
        'Source': source,
        'Headline': headline,
        'Sentiment': sentiment,
        'Sentiment Score': sentiment_score,
        'Label': label,
        'Link': link
    }

    return entry

def save_manual_entry_to_database(entry):
    """Saves the manual entry to the database."""
    try:
        conn = get_db_connection()
        if conn is None:
            return
        cur = conn.cursor()

        # Ensure the table exists with proper schema
        create_table_query = """
        CREATE TABLE IF NOT EXISTS crypto_news (
            date DATE,
            source VARCHAR(255),
            headline TEXT,
            sentiment VARCHAR(50),
            sentiment_score NUMERIC,
            label VARCHAR(50),
            link TEXT,
            UNIQUE (date, source, headline, link)
        );
        """
        cur.execute(create_table_query)
        conn.commit()

        # Insert data into the table
        insert_query = """
        INSERT INTO crypto_news (date, source, headline, sentiment, sentiment_score, label, link)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (date, source, headline, link) DO NOTHING;
        """
        cur.execute(insert_query, (
            entry['Date'],
            entry['Source'],
            entry['Headline'],
            entry['Sentiment'],
            entry['Sentiment Score'],
            entry['Label'],
            entry['Link']
        ))
        conn.commit()

        logging.info(f"Manual entry saved to the database.")
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Error saving manual entry to database: {e}")

def append_entry_to_csv(entry, csv_path):
    """Appends the manual entry to the specified CSV file."""
    try:
        df = pd.DataFrame([entry])
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, mode='w', header=True, index=False)
        logging.info(f"Manual entry appended to {csv_path}.")
    except Exception as e:
        logging.error(f"Error appending manual entry to CSV: {e}")

def main():
    """Main function to handle the manual news entry process."""
    entry = get_manual_entry()
    save_manual_entry_to_database(entry)

    # Append to main CSV file
    append_entry_to_csv(entry, MAIN_CSV_PATH)

    # If the headline or content mentions "DAO", append to DAO-specific CSV
    if 'dao' in entry['Headline'].lower():
        append_entry_to_csv(entry, DAO_CSV_PATH)

if __name__ == "__main__":
    main()
