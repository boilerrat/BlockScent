import logging
import pandas as pd
import feedparser
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from dateutil import parser
import json
import requests
import os
from newsapi import NewsApiClient
from components.utils import save_to_csv, save_to_database, get_db_connection, filter_headlines_by_keyword
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

def display_working_animation():
    animation_art = """
     ___           ___           ___           ___                       ___           ___     
     /\__\         /\  \         /\  \         /\__\          ___        /\__\         /\  \    
    /:/ _/_       /::\  \       /::\  \       /:/  /         /\  \      /::|  |       /::\  \   
   /:/ /\__\     /:/\:\  \     /:/\:\  \     /:/__/          \:\  \    /:|:|  |      /:/\:\  \  
  /:/ /:/ _/_   /:/  \:\  \   /::\~\:\  \   /::\__\____      /::\__\  /:/|:|  |__   /:/  \:\  \ 
 /:/_/:/ /\__\ /:/__/ \:\__\ /:/\:\ \:\__\ /:/\:::::\__\  __/:/\/__/ /:/ |:| /\__\ /:/__/_\:\__\\
 \:\/:/ /:/  / \:\  \ /:/  / \/_|::\/:/  / \/_|:|~~|~    /\/:/  /    \/__|:|/:/  / \:\  /\ \/__/
  \::/_/:/  /   \:\  /:/  /     |:|::/  /     |:|  |     \::/__/         |:/:/  /   \:\ \:\__\  
   \:\/:/  /     \:\/:/  /      |:|\/__/      |:|  |      \:\__\         |::/  /     \:\/:/  /  
    \::/  /       \::/  /       |:|  |        |:|  |       \/__/         /:/  /       \::/  /   
     \/__/         \/__/         \|__|         \|__|                     \/__/         \/__/    
    """
    print(animation_art)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load BERT model and tokenizer once
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Load NewsAPI key from environment
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize NewsApiClient
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def parse_rss_feed(name, url):
    """Parses the RSS feed and returns headlines."""
    feed = feedparser.parse(url)
    headlines = []
    for entry in feed.entries:
        title = entry.title
        link = entry.link
        content = entry.get('content', [{'value': None}])[0]['value'] or entry.get('summary', title)
        
        # Attempt to parse the date
        published = entry.get("published", None)
        if published:
            try:
                date = parser.parse(published, fuzzy=True).strftime('%Y-%m-%d')
            except (ValueError, parser.ParserError) as e:
                logging.error(f"Error parsing date for article from {name}: {title} - {e}")
                date = None
        else:
            logging.warning(f"Missing published date for article from {name}: {title}")
            date = None
        
        if date:
            headlines.append([date, name, title, content, link])
        else:
            logging.warning(f"Skipped article from {name} with unknown date: {title}")
            logging.debug(f"Full entry: {entry}")

    return headlines

def fetch_newsapi_articles(query=None, from_date=None, to_date=None, language='en'):
    """Fetches news articles using the NewsAPI client."""
    if from_date is None:
        # Set `from_date` to 30 days ago
        from_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
    if to_date is None:
        # Set `to_date` to today
        to_date = datetime.today().strftime('%Y-%m-%d')

    if query is None:
        # Default query to search for Bitcoin, Ethereum, Blockchain, and Crypto
        query = "Bitcoin OR Ethereum OR Blockchain OR Crypto"

    try:
        all_articles = newsapi.get_everything(
            q=query, 
            from_param=from_date, 
            to=to_date, 
            language=language, 
            sort_by='relevancy', 
            page_size=100
        )
        articles = []
        for article in all_articles['articles']:
            title = article['title']
            content = article['content'] or article['description']
            date = article['publishedAt']
            source = article['source']['name']
            link = article['url']
            if date:
                date = parser.parse(date).strftime('%Y-%m-%d')
            articles.append([date, source, title, content, link])
        return articles
    except Exception as e:
        logging.error(f"Failed to fetch NewsAPI data: {e}")
        return []

def analyze_sentiment(headlines):
    """Analyzes sentiment using BERT."""
    updated_headlines = []
    for headline in headlines:
        content = headline[3]
        inputs = tokenizer(content, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        sentiment_scores = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
        sentiment_label = sentiment_scores.argmax() + 1
        sentiment_score = round(float(sentiment_scores[sentiment_label - 1]), 4)
        sentiment = 'Positive' if sentiment_score > 0.5 else 'Negative'
        updated_headline = [headline[0], headline[1], headline[2], sentiment, sentiment_score, f"{sentiment_label} stars", headline[4]]
        updated_headlines.append(updated_headline)
    return updated_headlines

def fetch_and_process_sources(sources_file='components/sources.json'):
    """Fetches and processes all sources."""
    all_headlines = []
    with open(sources_file) as f:
        sources = json.load(f)["sources"]
    
    for source in sources:
        headlines = parse_rss_feed(source['name'], source['url'])
        if headlines:
            analyzed_headlines = analyze_sentiment(headlines)
            all_headlines.extend(analyzed_headlines)
    
    return pd.DataFrame(all_headlines, columns=["Date", "Source", "Headline", "Sentiment", "Sentiment Score", "Label", "Link"])

def get_historical_market_data(crypto_id, days=30):
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

def save_market_data_to_database(df, table_name):
    """Save market DataFrame to PostgreSQL database."""
    try:
        conn = get_db_connection()
        if conn is None:
            return

        cur = conn.cursor()

        # Ensure the table exists with proper schema
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

        # Replace NaN with None (interpreted as NULL in SQL)
        df = df.where(pd.notnull(df), None)

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

        logging.info(f"Market data saved to the {table_name} table.")
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Error saving market data to database: {e}")

def main():
    """Main function to fetch, process, and save data."""
    display_working_animation()  # Display ASCII art

    # Fetch and process RSS and NewsAPI sources
    df_headlines = fetch_and_process_sources()

    # Fetch additional news from NewsAPI
    newsapi_headlines = fetch_newsapi_articles(query="Cryptocurrency OR Bitcoin OR Ethereum OR Crypto OR Blockchain OR XRP")
    if newsapi_headlines:
        analyzed_newsapi_headlines = analyze_sentiment(newsapi_headlines)
        df_newsapi = pd.DataFrame(analyzed_newsapi_headlines, columns=["Date", "Source", "Headline", "Sentiment", "Sentiment Score", "Label", "Link"])
        df_headlines = pd.concat([df_headlines, df_newsapi], ignore_index=True)

    if not df_headlines.empty:
        save_to_csv(df_headlines, "csv/crypto_news_sentiment.csv")
        save_to_database(df_headlines, 'crypto_news')
        filter_headlines_by_keyword(df_headlines, "DAO", "csv/crypto_news_sentiment_DAO.csv")
    else:
        logging.warning("No headlines were scraped or fetched.")
    
    # Fetch market data
    btc_df = get_historical_market_data('bitcoin')
    eth_df = get_historical_market_data('ethereum')

    if not btc_df.empty and not eth_df.empty:
        merged_df = pd.merge(btc_df, eth_df, on='date', how='outer')
        save_market_data_to_database(merged_df, 'crypto_prices')
        save_to_csv(merged_df, 'csv/crypto_market_data.csv')
    else:
        logging.error("Failed to fetch data for Bitcoin or Ethereum.")

if __name__ == "__main__":
    main()
    logging.info("Script execution complete.")
