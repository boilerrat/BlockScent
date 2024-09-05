import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Create a connection to the database
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Retrieve sentiment data
sentiment_query = """
    SELECT date, sentiment_score
    FROM crypto_news;
"""
df_sentiment = pd.read_sql(sentiment_query, engine)

# Retrieve market data
market_query = """
    SELECT date, btc_price, eth_price
    FROM crypto_prices;
"""
df_market = pd.read_sql(market_query, engine)

# Close the engine connection
engine.dispose()

# Data Preprocessing
df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
df_market['date'] = pd.to_datetime(df_market['date'])

# Merge sentiment and market data on the date
df_combined = pd.merge(df_sentiment, df_market, on='date', how='inner')
df_combined.dropna(inplace=True)

# Correlation Analysis
correlation_btc, _ = pearsonr(df_combined['sentiment_score'], df_combined['btc_price'])
correlation_eth, _ = pearsonr(df_combined['sentiment_score'], df_combined['eth_price'])

print(f"Pearson correlation between sentiment scores and Bitcoin prices: {correlation_btc}")
print(f"Pearson correlation between sentiment scores and Ethereum prices: {correlation_eth}")

# Time Series Analysis - Visualizing
plt.figure(figsize=(14, 7))
plt.plot(df_combined['date'], df_combined['btc_price'], label='Bitcoin Price', color='blue')
plt.plot(df_combined['date'], df_combined['sentiment_score'], label='Sentiment Score', color='red')
plt.title('Bitcoin Price vs Sentiment Score Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(df_combined['date'], df_combined['eth_price'], label='Ethereum Price', color='green')
plt.plot(df_combined['date'], df_combined['sentiment_score'], label='Sentiment Score', color='red')
plt.title('Ethereum Price vs Sentiment Score Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Lag Analysis
df_combined['sentiment_score_lag'] = df_combined['sentiment_score'].shift(1)

correlation_btc_lag, _ = pearsonr(df_combined['sentiment_score_lag'].dropna(), df_combined['btc_price'][1:])
correlation_eth_lag, _ = pearsonr(df_combined['sentiment_score_lag'].dropna(), df_combined['eth_price'][1:])

print(f"Lagged Pearson correlation between sentiment scores and Bitcoin prices: {correlation_btc_lag}")
print(f"Lagged Pearson correlation between sentiment scores and Ethereum prices: {correlation_eth_lag}")

# Regression Analysis
X = df_combined[['sentiment_score_lag']].dropna()
y_btc = df_combined['btc_price'][1:]
y_eth = df_combined['eth_price'][1:]

X = sm.add_constant(X)  # Adds a constant term to the predictor

# Regression for Bitcoin
model_btc = sm.OLS(y_btc, X).fit()
print(model_btc.summary())

# Regression for Ethereum
model_eth = sm.OLS(y_eth, X).fit()
print(model_eth.summary())

# Granger Causality Test
max_lag = 5  # You can set this to a higher value depending on your needs

# Granger test for Bitcoin
print("\nGranger Causality Test - Bitcoin")
granger_btc = grangercausalitytests(df_combined[['btc_price', 'sentiment_score']], max_lag, verbose=True)

# Granger test for Ethereum
print("\nGranger Causality Test - Ethereum")
granger_eth = grangercausalitytests(df_combined[['eth_price', 'sentiment_score']], max_lag, verbose=True)
