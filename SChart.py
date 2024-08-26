import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv("crypto_news_sentiment2.csv")

# Ensure that the 'Date' column is a datetime object
df['Date'] = pd.to_datetime(df['Date'])

# Group by date and calculate the average sentiment score
average_sentiment = df.groupby('Date')['Sentiment Score'].mean()

# Plot the average sentiment score over time
plt.figure(figsize=(10, 6))
average_sentiment.plot(kind='line', marker='o')
plt.title('Average Sentiment Score Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("average_sentiment_score_over_time.png")

# Display the plot (optional)
plt.show()
