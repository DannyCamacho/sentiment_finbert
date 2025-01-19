import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the pre-trained FinBERT model
# model_name = "yiyanghkust/finbert-tone"
# sentiment_map = {"Negative": -1, "Neutral": 0, "Positive": 1}
model_name = "ProsusAI/finBERT"
sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set up the pipeline for sentiment analysis
finbert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def analyze_sentiment_finbert(text):
    result = finbert_pipeline(text)[0]
    return result['label'], result['score']


# Example usage
if __name__ == "__main__":
    stock_df = pd.read_csv("stockdata.csv")
    fngu_df = pd.read_csv("fngu_data.csv")
    fngu_df['diff'] = fngu_df['close'] - fngu_df['open']

    titles = stock_df['title']
    articles = stock_df['summary']
    dates = stock_df['published']
    dates_dict = {}

    for i in range(len(dates)):
        # Parse the input string to a datetime object
        date_obj = datetime.strptime(dates[i], "%a, %d %b %Y %H:%M:%S %z")
        formatted_date = date_obj.strftime("%Y-%m-%d")

        sentiment_label, confidence = analyze_sentiment_finbert(articles[i][:512])
        sentiment_label_t, confidence_t = analyze_sentiment_finbert(titles[i][:512])

        if formatted_date in dates_dict:
            dates_dict[formatted_date] = confidence * sentiment_map[sentiment_label] + dates_dict[formatted_date]
        else:
            dates_dict[formatted_date] = confidence * sentiment_map[sentiment_label]

    df_list = []

    for date in sorted(dates_dict.keys()):
        diff = fngu_df.loc[fngu_df['date'] == date]

        if diff['diff'].size > 0:
            df_list.append([date, diff['diff'].values.tolist()[0], dates_dict[date]])
            print(f"date: {date}, fngu diff: {diff['diff'].values.tolist()[0]}, sentiment score: {dates_dict[date]}")

    df = pd.DataFrame(df_list, columns=['date', 'diff', 'score'])
    # df['diff'] = (df['diff'] - df['diff'].mean()) / df['diff'].std()
    # df['score'] = (df['score'] - df['score'].mean()) / df['score'].std()
    df['diff'] = (df['diff'] - df['diff'].min()) / (df['diff'].max() - df['diff'].min())
    df['score'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())

    print(f"dot product: {df['diff'].dot(df['score'])}")
    print(f"mse: {((df['diff'] - df['score']) ** 2).mean()}")

    # Plot multiple line plots using only pandas
    df.plot(x='date', y=['diff', 'score'],
            kind='line',
            title='Line Plots of fngu diff and sentiment score')
    plt.title('Line Plots of FNGU Open/Close Diff and Combined Sentiment Score by Day')
    plt.xlabel('Dates')
    plt.xticks(rotation='vertical')
    plt.ylabel('FNGU Open/Close Diff and Combined Sentiment Score')
    plt.show()
