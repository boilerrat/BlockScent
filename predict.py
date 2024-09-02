import os
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def get_combined_sentiment_and_prices():
    """
    Queries the average sentiment scores for BTC and ETH, combines them into a single sentiment score,
    and retrieves the associated prices from the database.
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()

        # SQL query to get average sentiment scores and prices for each day
        query = """
        SELECT
            s.date,
            AVG(s.sentiment_score) AS avg_sentiment_score,
            p.eth_price,
            p.btc_price
        FROM
            crypto_news s
        JOIN
            crypto_prices p ON s.date = p.date
        GROUP BY
            s.date, p.eth_price, p.btc_price
        ORDER BY
            s.date ASC;
        """
        cur.execute(query)

        # Fetch all results into a DataFrame
        df = pd.DataFrame(cur.fetchall(), columns=['date', 'avg_sentiment_score', 'eth_price', 'btc_price'])

        # Calculate combined sentiment score
        df['combined_sentiment_score'] = df['avg_sentiment_score']

        # Convert types
        df['eth_price'] = df['eth_price'].astype(float)
        df['btc_price'] = df['btc_price'].astype(float)

        cur.close()
        conn.close()

        return df

    except Exception as e:
        print(f"Error: {e}")
        return None

def predict_prices_based_on_sentiment(df):
    """
    Predicts Ethereum and Bitcoin prices based on the combined sentiment score using linear regression.
    """
    # Train a linear regression model for Ethereum
    X_eth = df['combined_sentiment_score'].values.reshape(-1, 1)
    Y_eth = df['eth_price'].values.reshape(-1, 1)
    eth_model = LinearRegression().fit(X_eth, Y_eth)
    df['predicted_eth_price'] = eth_model.predict(X_eth)

    # Train a linear regression model for Bitcoin
    X_btc = df['combined_sentiment_score'].values.reshape(-1, 1)
    Y_btc = df['btc_price'].values.reshape(-1, 1)
    btc_model = LinearRegression().fit(X_btc, Y_btc)
    df['predicted_btc_price'] = btc_model.predict(X_btc)

    # Print model coefficients
    print(f"Ethereum Model Coefficient: {eth_model.coef_[0][0]:.4f}")
    print(f"Bitcoin Model Coefficient: {btc_model.coef_[0][0]:.4f}")

    # Calculate and print Mean Squared Error for the predictions
    eth_mse = mean_squared_error(df['eth_price'], df['predicted_eth_price'])
    btc_mse = mean_squared_error(df['btc_price'], df['predicted_btc_price'])
    print(f"Ethereum MSE: {eth_mse:.4f}")
    print(f"Bitcoin MSE: {btc_mse:.4f}")

    return df

def plot_predicted_vs_actual_prices(df):
    """
    Plots the predicted vs actual prices for Ethereum and Bitcoin.
    """
    plt.figure(figsize=(14, 7))
    
    # Ethereum
    plt.subplot(2, 1, 1)
    plt.plot(df['date'], df['eth_price'], label='Actual ETH Price', color='blue')
    plt.plot(df['date'], df['predicted_eth_price'], label='Predicted ETH Price', color='orange')
    plt.title('Ethereum: Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()

    # Bitcoin
    plt.subplot(2, 1, 2)
    plt.plot(df['date'], df['btc_price'], label='Actual BTC Price', color='green')
    plt.plot(df['date'], df['predicted_btc_price'], label='Predicted BTC Price', color='red')
    plt.title('Bitcoin: Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('/media/boilerrat/Bobby/CryptoData/BlockScent/charts/predicted_vs_actual_prices.png')
    plt.show()

if __name__ == "__main__":
    # Get combined sentiment and price data
    df = get_combined_sentiment_and_prices()

    if df is not None and not df.empty:
        # Predict prices based on sentiment
        df = predict_prices_based_on_sentiment(df)

        # Plot the predicted vs actual prices
        plot_predicted_vs_actual_prices(df)
    else:
        print("No data available for prediction.")
