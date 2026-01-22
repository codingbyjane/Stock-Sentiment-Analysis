# import ncessary components from transformers library
from transformers import pipeline, AutoTokenizer, set_seed

# import dataset library
from datasets import load_dataset

# Import data manipulation and plotting libraries
import pandas as pd
import matplotlib.pyplot as plt
from venny4py.venny4py import venny4py

# Import date manipulation classes
from datetime import datetime, timedelta

# Import regular expression module for parsing dates
import re

# Import yfinance module for accessing stock price data
import yfinance as yf

# Load XSum dataset
dataset = load_dataset('xsum')

# Filter full (hence, applied 'document' not 'summary') articles that mention 'Tesla'
tesla_articles = dataset['train'].filter(lambda article: 'Tesla' in article['document']) # Applying a lambda inside a higher order function to filter dataset

# Convert filtered dataset to pandas DataFrame
tesla_df = pd.DataFrame(tesla_articles)

#print(tesla_df.head(10))
print(f"Total articles mentioning Tesla: {len(tesla_df)}") # Total articles mentioning Tesla: 210

# Initialize sentiment analysis pipeline
def text_classification(model_name, dataset):
    # Ensure reproducibility
    set_seed(42)

    # Initialize the sentiment analysis pipeline
    pipe = pipeline("text-classification", model=model_name)

    # Analyze texts sentiment ("document" field of the Xsum dataset)
    result = pipe(list(dataset['document']), truncation=True, max_length = 512) # Truncation and max_length to handle long articles
    # Clean up resources
    del pipe # Deleting pipe from object memrory to clean up GPU/CPU space

    return pd.DataFrame(result)

# Perform sentiment analysis on Tesla articles using dffferent models
finbert_result = text_classification("ProsusAI/finbert", tesla_articles)
distilbert_result = text_classification("distilbert/distilbert-base-uncased", tesla_articles)
bert_base_result = text_classification("bert-base-uncased", tesla_articles)
deepseek_result = text_classification("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", tesla_articles)

# Harmonize sentiment labels across the models for comparison
bert_base_result['label'].replace(['1 star', '2 stars'], 'NEGATIVE', inplace=True) # Inplace=True modifies the existing DataFrame instead of creating a new one
bert_base_result['label'].replace(['4 star', '5 stars'], 'POSITIVE', inplace=True)
bert_base_result['label'].replace(['3 stars'], 'NEUTRAL', inplace=True)

distilbert_result['label'].replace(['LABEL_0'], 'NEGATIVE', inplace=True)
distilbert_result['label'].replace(['LABEL_1'], 'POSITIVE', inplace=True)

deepseek_result['label'].replace(['LABEL_0'], 'NEGATIVE', inplace=True)
deepseek_result['label'].replace(['LABEL_1'], 'POSITIVE', inplace=True)

# Plot negative sentiment distribution overlap for each model
finbert_negative = finbert_result[finbert_result['label'] == 'NEGATIVE'].index.tolist()
distilbert_negative = distilbert_result[distilbert_result['label'] == 'NEGATIVE'].index.tolist()
bert_base_negative = bert_base_result[bert_base_result['label'] == 'NEGATIVE'].index.tolist()
deepseek_negative = deepseek_result[deepseek_result['label'] == 'NEGATIVE'].index.tolist()

sets_negative = {
    "FinBERT": set(finbert_negative),
    "DistilBERT": set(distilbert_negative),
    "BERT-Base": set(bert_base_negative),
    "DeepSeek": set(deepseek_negative)
}

# Plot Venn diagram for negative sentiment overlap
venny4py(sets = sets_negative)
plt.title("Negative Sentiment Overlap Between Models")

# Plot positive sentiment distribution overlap for each model using the same process
finbert_positive = finbert_result[finbert_result['label'] == 'POSITIVE'].index.tolist()
distilbert_positive = distilbert_result[distilbert_result['label'] == 'POSITIVE'].index.tolist()
bert_base_positive = bert_base_result[bert_base_result['label'] == 'POSITIVE']. index.tolist()
deepseek_positive = deepseek_result[deepseek_result['label'] == 'POSITIVE'].index.tolist()

sets_positive = {
    "FinBERT": set(finbert_positive),
    "DistilBERT": set(distilbert_positive),
    "BERT-Base": set(bert_base_positive),
    "DeepSeek": set(deepseek_positive)
}

# Plot Venn diagram for positive sentiment overlap
venny4py(sets = sets_positive)
plt.title("Positive Sentiment Overlap Between Models")

plt.show()

