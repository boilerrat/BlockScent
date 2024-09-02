import logging
import pandas as pd
import feedparser
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from dateutil import parser
from components.utils import save_to_csv, save_to_database
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def parse_rss_feed(name, url):
    try:
        feed = feedparser.parse(url)
        headlines = []
        for entry in feed.entries:
            title = entry.title
            link = entry.link

            content = entry.get('content', [{'value': None}])[0]['value']
            if not content:
                content = entry.get('summary', title)

            published = entry.get("published", "Unknown")
            date = "Unknown"
            if published != "Unknown":
                try:
                    # Try multiple formats for date parsing
                    date = parser.parse(published, fuzzy=True).strftime('%Y-%m-%d')
                except (ValueError, parser.ParserError) as e:
                    logging.error(f"Error parsing date: {published} - {e}")
                    continue
            else:
                # Log articles with missing dates
                logging.warning(f"Skipped article with unknown date: {title}")
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
        content = headline[3]
        inputs = tokenizer(content, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        sentiment_scores = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
        sentiment_label = sentiment_scores.argmax() + 1  # 1-5 stars
        sentiment_score = round(float(sentiment_scores[sentiment_label - 1]), 4)  # Round score to 4 decimal places
        sentiment = 'Positive' if sentiment_score > 0.5 else 'Negative'
        logging.debug(f"Sentiment: {sentiment}, Score: {sentiment_score}, Label: {sentiment_label} stars")

        updated_headline = [headline[0], headline[1], headline[2], sentiment, sentiment_score, f"{sentiment_label} stars", headline[4]]
        updated_headlines.append(updated_headline)
        
    return updated_headlines

def fetch_and_process_sources():
    all_headlines = []
    with open('components/sources.json') as f:
        sources = json.load(f)["sources"]

    for source in sources:
        try:
            headlines = parse_rss_feed(source['name'], source['url'])
            if headlines:
                analyzed_headlines = analyze_sentiment(headlines)
                all_headlines.extend(analyzed_headlines)
            else:
                logging.warning(f"No headlines were scraped from {source['name']}.")
        except Exception as e:
            logging.error(f"Failed to scrape {source['name']}: {e}")

    return pd.DataFrame(all_headlines, columns=["Date", "Source", "Headline", "Sentiment", "Sentiment Score", "Label", "Link"])

def filter_headlines_by_keyword(df, keyword, output_filename):
    filtered_df = df[df['Headline'].str.contains(keyword, case=False, na=False)]
    
    if not filtered_df.empty:
        save_to_csv(filtered_df, output_filename)
        logging.info(f"Filtered data containing keyword '{keyword}' saved to {output_filename}")
    else:
        logging.warning(f"No headlines found containing the keyword '{keyword}'")

if __name__ == "__main__":
    df_headlines = fetch_and_process_sources()

    if not df_headlines.empty:
        save_to_csv(df_headlines, "csv/crypto_news_sentiment2.csv")
        save_to_database(df_headlines, 'crypto_news')
        filter_headlines_by_keyword(df_headlines, "DAO", "csv/crypto_news_sentiment_DAO.csv")
    else:
        logging.warning("No headlines were scraped.")

    logging.info("Scraping complete.")
