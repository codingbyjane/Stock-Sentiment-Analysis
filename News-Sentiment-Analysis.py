# import ncessary components from transformers library
from transformers import pipline, AutoTokenizer, set_seed

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