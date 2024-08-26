import requests
from bs4 import BeautifulSoup
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

def scrape_coindesk():
    url = "https://www.coindesk.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    headlines = []
    for article in soup.find_all('a', class_='heading'):
        title = article.get_text().strip()
        link = article['href']
        headlines.append([title, link])

    df = pd.DataFrame(headlines, columns=["Title", "Link"])
    return df

def analyze_sentiment(df):
    sid = SentimentIntensityAnalyzer()
    df["Sentiment"] = df["Title"].apply(lambda title: sid.polarity_scores(title)["compound"])
    return df

def save_to_csv(df, filename="news_sentiment.csv"):
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    news_df = scrape_coindesk()
    sentiment_df = analyze_sentiment(news_df)
    save_to_csv(sentiment_df)
    print("News headlines and sentiment data saved to news_sentiment.csv")
