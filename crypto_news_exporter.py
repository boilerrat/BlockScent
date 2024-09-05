import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

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

# Query to select all data from the crypto_news table
query = "SELECT * FROM crypto_news;"

# Read the data into a pandas DataFrame
df_crypto_news = pd.read_sql(query, engine)

# Close the engine connection
engine.dispose()

# Define the output file path
output_file = 'csv/crypto_news_export.csv'

# Save the DataFrame to a CSV file
df_crypto_news.to_csv(output_file, index=False)

print(f"Data successfully exported to {output_file}")
