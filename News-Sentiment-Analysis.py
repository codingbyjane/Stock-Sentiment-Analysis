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
print(f"Total articles mentioning Tesla: {len(tesla_df)}")

# Initialize sentiment analysis pipeline
def text_classification(model_name, text):
    # Ensure reproducibility
    set_seed(42)

    # Initialize the sentiment analysis pipeline
    pipe = pipeline("text-classification", model=model_name)

    # Analyze texts sentiment
    result = pipe(text['article'], truncation=True, max_length = 512)

    # Clean up resources
    del pipe

    return pd.DataFrame(result)