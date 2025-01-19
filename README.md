# FinBERT Sentiment Analysis and Stock Data Visualization

This repository contains a Python script that utilizes the FinBERT model from Hugging Face's `transformers` library to perform sentiment analysis on financial news articles. The sentiment scores are then compared with stock price differences from the FNGU stock index.

## Overview

The script is designed to:

- Analyze sentiment of financial news articles using FinBERT.
- Compare the sentiment scores to the stock price differences (open vs. close) from the FNGU index.
- Normalize and plot these scores to visualize their relationship.

## Components

### Sentiment Analysis

The script uses the following components for sentiment analysis:

- **Model**: FinBERT (from Hugging Face)
- **Tokenizer**: FinBERT tokenizer
- **Pipeline**: Hugging Face's sentiment-analysis pipeline

### Data Processing

- **Input Files**: 
  - `stockdata.csv`: Contains columns `title`, `summary`, and `published` (date).
  - `fngu_data.csv`: Contains columns `date`, `open`, and `close`.
- **Sentiment Calculation**: Sentiment scores are calculated for each news article and title.
- **Stock Difference Calculation**: The difference between open and close prices is calculated for each day.

### Plotting

- The script plots the normalized sentiment scores and stock price differences using `matplotlib`.

## Model Details

### FinBERT Sentiment Analysis

- **Model Name**: ProsusAI/finBERT
- **Sentiment Mapping**:
  - Positive: 1
  - Neutral: 0
  - Negative: -1

### Stock Data Processing

- Calculates the daily difference between the open and close prices from `fngu_data.csv`.
- Aggregates sentiment scores by date to compare with stock differences.

## Training and Testing

### Sentiment Analysis

- Sentiment scores are derived from the FinBERT model for each article.
- Confidence scores for the sentiment are calculated and mapped to numeric values.

### Stock Data Analysis

- The script computes the difference between open and close prices.
- These differences are then normalized and compared with the sentiment scores.

## Example Output

The script prints the date-wise sentiment scores and stock differences, calculates their dot product, and the mean squared error (MSE):

![image](https://github.com/user-attachments/assets/8d4f2607-1827-4200-9c92-a4442597f48b)
