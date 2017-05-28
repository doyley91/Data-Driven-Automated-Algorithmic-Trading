import pandas as pd

import functions as fc
from mlearning import SGDr

df_corr = pd.read_csv("data/WIKI_PRICES_corr.csv", index_col='ticker')

stocks = fc.get_neutrally_correlated_stocks(df_corr, correlation=0.1)

ticker = "MSFT"

tickers = fc.get_stocks_from_list(stocks, ticker)

for symbol in tickers:
    SGDr.main(symbol)
