import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv("crypto_news_sentiment2.csv")

# Ensure that the 'Date' column is a datetime object
df['Date'] = pd.to_datetime(df['Date'])

# Group by date and calculate the overall average sentiment score across all sources
overall_daily_sentiment = df.groupby('Date')['Sentiment Score'].mean()

# Plot the overall daily sentiment score over time
plt.figure(figsize=(10, 6))
overall_daily_sentiment.plot(kind='line', marker='o')
plt.title('Overall Daily Sentiment Score Over Time')
plt.xlabel('Date')
plt.ylabel('Overall Average Sentiment Score')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("overall_daily_sentiment_score.png")

# Display the plot (optional)
plt.show()

# Save the overall daily sentiment score to a new CSV file
overall_daily_sentiment.to_csv("overall_daily_sentiment_score.csv", header=["Sentiment Score"])
