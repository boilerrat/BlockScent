SELECT 
    date,
    COUNT(*) AS total_articles,
    AVG(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) AS average_sentiment, 
    AVG(sentiment_score) AS average_sentiment_score,
    AVG(CASE 
        WHEN label = '1 stars' THEN 1 
        WHEN label = '2 stars' THEN 2 
        WHEN label = '3 stars' THEN 3 
        WHEN label = '4 stars' THEN 4 
        WHEN label = '5 stars' THEN 5 
    END) AS average_star_label
FROM crypto_news
WHERE date = '2024-08-27' -- Change this
GROUP BY date;