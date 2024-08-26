# Crypto News Sentiment Analyzer

## Overview

The Crypto News Sentiment Analyzer is a tool designed to provide sentiment analysis of cryptocurrency-related news headlines. By scraping headlines from various news sources via RSS feeds, analyzing them using BERT (Bidirectional Encoder Representations from Transformers), and storing the results in a CSV file, this tool helps users understand the overall mood of the cryptocurrency market as reflected in the news.

## Features

- **RSS Feed Parsing**: The tool pulls headlines and article content from multiple cryptocurrency news sources using RSS feeds.
- **Sentiment Analysis**: Each headline is analyzed using the BERT model, which assigns a sentiment label (ranging from 1 to 5 stars) and a confidence score based on the content of the entire article.
- **CSV Output**: The analyzed data is saved in a CSV file, including the publication date, headline, link, source, sentiment label, and sentiment score.
- **Data Aggregation**: The script calculates the average sentiment score across all headlines to provide an overall sentiment snapshot.
- **Duplicate Handling**: The tool is designed to avoid counting the same story from the same source multiple times.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/crypto-news-sentiment-analyzer.git
cd crypto-news-sentiment-analyzer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Sources

- Modify the `sources.json` file located in the `components/` directory to include the RSS feeds of your preferred news sources.

### 4. Run the Script

```bash
python analyze_sentiment.py
```

## Usage

### Running the Script

- Simply run the script using the command above. The script will parse the RSS feeds, analyze the sentiment of each headline, and save the results in a CSV file (`crypto_news_sentiment.csv`).

### Scheduling

- To keep your sentiment analysis data up-to-date, you can schedule the script to run periodically using a task scheduler like `cron` on Linux or `Task Scheduler` on Windows.

## Explanation of Sentiment Analysis

### Understanding BERT's Sentiment Scores

BERT (Bidirectional Encoder Representations from Transformers) is a powerful model developed by Google for NLP tasks. In this project, we use a pre-trained BERT model fine-tuned for sentiment analysis, which outputs a sentiment score and label for each article.

- **Sentiment Label (Stars)**: The BERT model outputs a sentiment label ranging from 1 to 5 stars:
  - 1 Star: Very Negative
  - 2 Stars: Negative
  - 3 Stars: Neutral
  - 4 Stars: Positive
  - 5 Stars: Very Positive

- **Sentiment Score**: The sentiment score is a confidence score between 0 and 1, representing how strongly the model feels about its assigned sentiment label. 
  - A score closer to 1 indicates high confidence in the sentiment label.
  - A score closer to 0.5 indicates less confidence, meaning the sentiment could be more ambiguous.

### Combined Sentiment Interpretation

To provide a more nuanced understanding of the sentiment:
- **Positive/Negative Label**: This label is determined by whether the sentiment score is greater than or less than 0.5. 
  - **Positive**: A sentiment score greater than 0.5.
  - **Negative**: A sentiment score less than or equal to 0.5.

### Data Columns

- **Date**: The publication date of the article, extracted from the RSS feed. If the date is unavailable, it is listed as "Unknown."
- **Headline**: The title of the article as provided by the RSS feed.
- **Sentiment**: A label indicating whether the overall sentiment is Positive or Negative, based on the sentiment score.
- **Stars**: The sentiment label assigned by the BERT model, represented as a rating from 1 to 5 stars.
- **Score**: The sentiment confidence score, ranging from 0 to 1, with higher values indicating stronger confidence in the sentiment label.
- **Link**: The URL to the original article.

## Roadmap

### Short-Term Goals

1. **Expand Data Sources**:
   - Add more RSS feeds from additional cryptocurrency news websites to improve the breadth of sentiment analysis.

2. **Enhanced Sentiment Scoring**:
   - Implement more nuanced sentiment analysis by considering contextual word meanings and additional NLP techniques.

3. **Duplicate Handling**:
   - Improve functionality to ensure that the same story from the same source is not counted multiple times.

### Mid-Term Goals

1. **Database Integration**:
   - Store results in a database (e.g., SQLite, PostgreSQL) for better data management and querying capabilities.

2. **Scheduled Runs**:
   - Set up the script to run on a schedule (e.g., daily) to continuously update the sentiment analysis data.

### Long-Term Goals

1. **Web Interface**:
   - Develop a simple web interface where users can view sentiment trends over time, filter by date or source, and export data as needed.

2. **Real-Time Sentiment Analysis**:
   - Implement real-time sentiment analysis to provide up-to-the-minute sentiment insights.

3. **Sentiment Trends & Visualizations**:
   - Create visualizations of sentiment trends over time to better understand the market's mood.

4. **VPS Deployment**:
   - Deploy the entire system on a Virtual Private Server (VPS) to ensure it's always running and accessible.

