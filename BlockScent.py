import logging
import pandas as pd
import feedparser
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
from datetime import datetime

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

            # Attempt to use full content, then summary, then title
            content = entry.get('content', [{'value': None}])[0]['value']
            if not content:
                content = entry.get('summary', title)

            published = entry.get("published", "Unknown")
            # Extract date only
            if published != "Unknown":
                date = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %z').strftime('%Y-%m-%d')
            else:
                date = "Unknown"
                
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
    for headline in headlines:
        content = headline[3]  # Content is at index 3
        inputs = tokenizer(content, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        sentiment_scores = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
        sentiment_label = sentiment_scores.argmax() + 1  # 1-5 stars
        sentiment_score = sentiment_scores[sentiment_label - 1]
        sentiment = 'Positive' if sentiment_score > 0.5 else 'Negative'
        headline[3] = sentiment  # Replace content with sentiment (Positive/Negative)
        headline.insert(4, sentiment_score)  # Insert sentiment score after sentiment
        headline.insert(5, f"{sentiment_label} stars")  # Insert sentiment label after sentiment score
    return headlines

def save_to_csv(all_headlines, filename="crypto_news_sentiment2.csv"):
    # Convert to DataFrame with the desired column order
    df = pd.DataFrame(all_headlines, columns=["Date", "Source", "Sentiment", "Sentiment Score", "Label", "Headline", "Link"])
    
    # Ensure the Sentiment Score column is numeric
    df['Sentiment Score'] = pd.to_numeric(df['Sentiment Score'], errors='coerce')
    
    # Save to CSV
    df.to_csv(filename, index=False)
    logging.info(f"Data saved to {filename}")

if __name__ == "__main__":
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
        save_to_csv(all_headlines)
    else:
        logging.warning("No headlines were scraped.")

    logging.info("Scraping complete.")
