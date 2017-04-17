import functions as fc
from mlearning import sgdregressor
import pandas as pd

df_corr = pd.read_csv("data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db_corr.csv", index_col='ticker')

stocks = fc.get_neutrally_correlated_stocks(df_corr, correlation=0.1)

ticker = "MSFT"

tickers = fc.get_stocks_from_list(stocks, ticker)

for symbol in tickers:
    sgdregressor.ticker = symbol
