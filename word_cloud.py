import pandas as pd
from sqlalchemy import create_engine
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import re

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

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, max_words=150, background_color='white').generate(all_text)

# Plotting the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top 100 Words in Crypto News Headlines', fontsize=20)
plt.show()


# Ensure the /charts/ directory exists
output_dir = 'charts/'
os.makedirs(output_dir, exist_ok=True)

# Save the word cloud image with a unique name based on the date
output_image = f"{output_dir}crypto_news_wordcloud_150.png"
wordcloud.to_file(output_image)

# Save the word cloud image
output_image = 'charts/crypto_news_wordcloud.png'
wordcloud.to_file(output_image)

print(f"Word cloud image saved as {output_image}")
