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

# Filter articles that mention 'Tesla'
tesla_articles = dataset['train'].filter(lambda example: 'Tesla' in example['document'])

# Convert filtered dataset to pandas DataFrame
tesla_df = pd.DataFrame(tesla_articles)

#print(tesla_df.head(10))
print(f"Total articles mentioning Tesla: {len(tesla_df)}") # Total articles mentioning Tesla: 210

# Initialize sentiment analysis pipeline
def text_classification(model_name, text):
    # Ensure reproducibility
    set_seed(42)

    # Initialize the sentiment analysis pipeline
    pipe = pipeline("text-classification", model=model_name)

    # Analyze texts sentiment
    result = pipe(text['article'], truncation=True, max_length = 512, batch_zize=16) # adding a batch size for faster processing

    # Clean up resources
    del pipe # Deleting pipe from object memrory to clean up GPU/CPU space

    return pd.DataFrame(result)

# Perform sentiment analysis on Tesla articles using dffferent models
finbert_result = text_classification("ProsusAI/finbert", tesla_articles)
distilbert_result = text_classification("distilbert/distilbert-base-uncased", tesla_articles)
bert_base_result = text_classification("bert-base-uncased", tesla_articles)
deepseek_result = text_classification("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", tesla_articles)

# Harmonize sentiment labels across the models for comparison
bert_base_result['label'].replace(['1 star', '2 stars'], 'NEGATIVE', inplace=True)
bert_base_result['label'].replace(['4 star', '5 stars'], 'POSITIVE', inplace=True)
bert_base_result['label'].replace(['3 stars'], 'NEUTRAL', inplace=True)

distilbert_result['label'].replace(['LABEL_0'], 'NEGATIVE', inplace=True)
distilbert_result['label'].replace(['LABEL_1'], 'POSITIVE', inplace=True)

deepseek_result['label'].replace(['LABEL_0'], 'NEGATIVE', inplace=True)
deepseek_result['label'].replace(['LABEL_1'], 'POSITIVE', inplace=True)

# Plot negative sentiment distribution overlap for each model
finbert_negative = finbert_result[finbert_result['label'] == 'NEGATIVE'].index.tolist()
distilbert_negative = distilbert_result[distilbert_result['label'] == 'NEGATIVE'].index.tolist()
bert_base_negative = bert_base_result[bert_base_result['lable'] == 'NEGATIVE'].index.tolist()
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
#plt.show()