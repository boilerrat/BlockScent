# data.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from components.utils import get_db_connection
from decimal import Decimal

# Directory to save charts
charts_dir = "/media/boilerrat/Bobby/CryptoData/BlockScent/charts"

def get_sentiment_vs_crypto_data():
    try:
        conn = get_db_connection()
        if conn is None:
            return None
        cur = conn.cursor()

        query = """
        SELECT
            s.date,
            AVG(s.sentiment_score) AS average_sentiment_score,
            p.eth_price,
            p.eth_market_cap,
            p.eth_volume,
            p.btc_price,
            p.btc_market_cap,
            p.btc_volume
        FROM
            crypto_news s
        JOIN
            crypto_market_data p ON s.date = p.date
        GROUP BY
            s.date, p.eth_price, p.eth_market_cap, p.eth_volume, p.btc_price, p.btc_market_cap, p.btc_volume
        ORDER BY
            s.date ASC;
        """
        cur.execute(query)
        df = pd.DataFrame(cur.fetchall(), columns=['date', 'average_sentiment_score', 'eth_price', 'eth_market_cap', 'eth_volume', 'btc_price', 'btc_market_cap', 'btc_volume'])

        df['average_sentiment_score'] = df['average_sentiment_score'].apply(float)
        df['eth_price'] = df['eth_price'].apply(float)
        df['eth_market_cap'] = df['eth_market_cap'].apply(float)
        df['eth_volume'] = df['eth_volume'].apply(float)
        df['btc_price'] = df['btc_price'].apply(float)
        df['btc_market_cap'] = df['btc_market_cap'].apply(float)
        df['btc_volume'] = df['btc_volume'].apply(float)

        cur.close()
        conn.close()

        return df

    except Exception as e:
        print(f"Error: {e}")
        return None

def plot_sentiment_vs_crypto(df, crypto, price_column, market_cap_column, volume_column):
    sns.set(style="whitegrid")

    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Average Sentiment Score', color='tab:blue')
    ax1.plot(df['date'], df['average_sentiment_score'], color='tab:blue', label='Avg Sentiment Score')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel(f'{crypto} Price (USD)', color='tab:orange')
    ax2.plot(df['date'], df[price_column], color='tab:orange', label=f'{crypto} Price')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.suptitle(f'Average Sentiment Score vs. {crypto} Price Over Time')
    fig.tight_layout()
    plt.savefig(os.path.join(charts_dir, f'sentiment_vs_{crypto.lower()}_price.png'))
    plt.show()

    plt.figure(figsize=(14, 7))
    sns.lineplot(x=df['date'], y=df['average_sentiment_score'], label='Avg Sentiment Score', color='blue')
    sns.lineplot(x=df['date'], y=np.log(df[market_cap_column]), label=f'{crypto} Market Cap (Log Scale)', color='green')
    plt.title(f'Average Sentiment Score vs. {crypto} Market Cap Over Time (Log Scale)')
    plt.xlabel('Date')
    plt.ylabel(f'{crypto} Market Cap (Log USD)')
    plt.legend()
    plt.savefig(os.path.join(charts_dir, f'sentiment_vs_{crypto.lower()}_market_cap_log.png'))
    plt.show()

    plt.figure(figsize=(14, 7))
    sns.lineplot(x=df['date'], y=df['average_sentiment_score'], label='Avg Sentiment Score', color='blue')
    sns.lineplot(x=df['date'], y=np.log(df[volume_column]), label=f'{crypto} Volume (Log Scale)', color='red')
    plt.title(f'Average Sentiment Score vs. {crypto} Volume Over Time (Log Scale)')
    plt.xlabel('Date')
    plt.ylabel(f'{crypto} Volume (Log USD)')
    plt.legend()
    plt.savefig(os.path.join(charts_dir, f'sentiment_vs_{crypto.lower()}_volume_log.png'))
    plt.show()

    print(f"Statistical Analysis for {crypto}:")

    for column in [price_column, market_cap_column, volume_column]:
        corr, _ = pearsonr(df['average_sentiment_score'], df[column])
        print(f"\nPearson correlation coefficient between sentiment and {crypto} {column}: {corr:.4f}")

        X = df['average_sentiment_score'].values.reshape(-1, 1)
        Y = df[column].values.reshape(-1, 1)
        model = LinearRegression().fit(X, Y)
        print(f"Linear regression coefficient between sentiment and {crypto} {column}: {model.coef_[0][0]:.4f}")

        plt.figure(figsize=(8, 6))
        sns.regplot(x=df['average_sentiment_score'], y=df[column])
        plt.title(f'Linear Regression: Sentiment Score vs {crypto} {column}')
        plt.xlabel('Average Sentiment Score')
        plt.ylabel(f'{crypto} {column} (USD)')

        plt.text(0.05, 0.95, f'Pearson r: {corr:.4f}\nRegression Coeff: {model.coef_[0][0]:.4f}', 
                 ha='left', va='center', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.5))

        plt.savefig(os.path.join(charts_dir, f'sentiment_vs_{crypto.lower()}_{column}_regression.png'))
        plt.show()

if __name__ == "__main__":
    df = get_sentiment_vs_crypto_data()

    if df is not None and not df.empty:
        plot_sentiment_vs_crypto(df, 'Ethereum', 'eth_price', 'eth_market_cap', 'eth_volume')
        plot_sentiment_vs_crypto(df, 'Bitcoin', 'btc_price', 'btc_market_cap', 'btc_volume')
    else:
        print("No data available to plot.")
