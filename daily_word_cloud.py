import pandas as pd
from sqlalchemy import create_engine
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import re
from datetime import datetime, timedelta

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

# Calculate the previous day's date
yesterday = datetime.now() - timedelta(1)
yesterday_str = yesterday.strftime('%Y-%m-%d')

# Query to select all data from the crypto_news table for the previous day
query = f"SELECT * FROM crypto_news WHERE date = '{yesterday_str}';"

# Read the data into a pandas DataFrame
df_crypto_news = pd.read_sql(query, engine)

# Close the engine connection
engine.dispose()

# If there are no headlines from the previous day, exit the script
if df_crypto_news.empty:
    print(f"No headlines found for {yesterday_str}. Exiting.")
    exit()

# Preprocessing the Headline column to extract words
def preprocess_text(text):
    # Remove special characters, numbers, and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# Apply preprocessing to the 'Headline' column
df_crypto_news['processed_headline'] = df_crypto_news['headline'].apply(preprocess_text)

# Combine all headlines into a single string
all_text = ' '.join(df_crypto_news['processed_headline'].tolist())

# Generate a word cloud limited to 15 words
wordcloud = WordCloud(width=800, height=400, max_words=30, background_color='white').generate(all_text)

# Plotting the word cloud (no title or heading)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# Ensure the /charts/ directory exists
output_dir = 'charts/'
os.makedirs(output_dir, exist_ok=True)

# Save the word cloud image with a unique name based on the date
output_image = f"{output_dir}crypto_news_wordcloud_{yesterday_str}.png"
wordcloud.to_file(output_image)

print(f"Word cloud image for {yesterday_str} saved as {output_image}")
