import logging
import pandas as pd
import feedparser
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Database connection
conn = psycopg2.connect(
    dbname="crypto_data",
    user="cryptouser",
    password="your_secure_password",
    host="localhost"
)

def create_table():
    with conn.cursor() as cur:
        cur.execute('''
        CREATE TABLE IF NOT EXISTS crypto_sentiment (
            id SERIAL PRIMARY KEY,
            date DATE NOT NULL,
            source TEXT NOT NULL,
            headline TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            sentiment_score REAL NOT NULL,
            label TEXT NOT NULL,
            link TEXT NOT NULL,
            UNIQUE (headline, link)
        );
        ''')
        conn.commit()

def parse_rss_feed(name, url):
    try:
        feed = feedparser.parse(url)
        headlines = []
        for entry in feed.entries:
            title = entry.title
            link = entry.link

            # Attempt to use full content, then summary, then title
            content = entry.get('content', [{'value': None}])[0]['value']
            if not content:
                content = entry.get('summary', title)

            published = entry.get("published", "Unknown")
            date = "Unknown"
            # Attempt to extract date with and without timezone information
            try:
                if published != "Unknown":
                    date = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %z').strftime('%Y-%m-%d')
            except ValueError:
                try:
                    date = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S').strftime('%Y-%m-%d')
                except ValueError:
                    logging.error(f"Error parsing date: {published}")
                    continue
                
            logging.debug(f"{name} - Article Found: {title}, Link: {link}")
            headlines.append([date, name, title, content, link])  # Store date, source, title, content, and link
        if headlines:
            logging.info(f"Scraped {len(headlines)} headlines from {name}.")
        else:
            logging.warning(f"No headlines were scraped from {name}.")
        return headlines
    except Exception as e:
        logging.error(f"Error parsing RSS feed from {name}: {e}")
        return []

def analyze_sentiment(headlines):
    updated_headlines = []
    for headline in headlines:
        content = headline[3]  # Content is at index 3
        inputs = tokenizer(content, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        sentiment_scores = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
        sentiment_label = sentiment_scores.argmax() + 1  # 1-5 stars
        sentiment_score = round(float(sentiment_scores[sentiment_label - 1]), 4)  # Round score to 4 decimal places
        sentiment = 'Positive' if sentiment_score > 0.5 else 'Negative'
        logging.debug(f"Sentiment: {sentiment}, Score: {sentiment_score}, Label: {sentiment_label} stars")
        
        # Replace content with sentiment, and insert score and label
        updated_headline = [headline[0], headline[1], headline[2], sentiment, sentiment_score, f"{sentiment_label} stars", headline[4]]
        updated_headlines.append(updated_headline)
        
    return updated_headlines

def save_to_database(all_headlines):
    with conn.cursor() as cur:
        insert_query = '''
        INSERT INTO crypto_sentiment (date, source, headline, sentiment, sentiment_score, label, link)
        VALUES %s
        ON CONFLICT (headline, link) DO NOTHING;
        '''
        execute_values(cur, insert_query, all_headlines)
        conn.commit()
    logging.info(f"Data saved to the database.")

def filter_headlines_by_keyword(all_headlines, keyword, output_filename):
    # Filter headlines containing the keyword (case-insensitive)
    filtered_headlines = [headline for headline in all_headlines if keyword.lower() in headline[2].lower()]
    
    if filtered_headlines:
        # Convert to DataFrame with the desired column order
        df = pd.DataFrame(filtered_headlines, columns=["Date", "Source", "Headline", "Sentiment", "Sentiment Score", "Label", "Link"])
        
        # Save to CSV
        df.to_csv(output_filename, index=False)
        logging.info(f"Filtered data containing keyword '{keyword}' saved to {output_filename}")
    else:
        logging.warning(f"No headlines found containing the keyword '{keyword}'")

if __name__ == "__main__":
    create_table()
    all_headlines = []

    # Load the JSON configuration file
    with open('components/sources.json') as f:
        sources = json.load(f)["sources"]

    for source in sources:
        try:
            headlines = parse_rss_feed(source['name'], source['url'])
            analyzed_headlines = analyze_sentiment(headlines)
            all_headlines.extend(analyzed_headlines)
        except Exception as e:
            logging.error(f"Failed to scrape {source['name']}: {e}")

    if all_headlines:
        save_to_database(all_headlines)
        # Filter and save headlines containing the keyword "DAO"
        filter_headlines_by_keyword(all_headlines, "DAO", "crypto_news_sentiment_DAO.csv")
    else:
        logging.warning("No headlines were scraped.")

    logging.info("Scraping complete.")
