from collections import OrderedDict

import pandas as pd
import pytz
from zipline.algorithm import TradingAlgorithm

import functions as fc
from strategies import buy_and_hold

df_corr = pd.read_csv("data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db_corr.csv", index_col='ticker')

stocks = fc.get_neutrally_correlated_stocks(df_corr, correlation=0.1)

ticker = "MSFT"

tickers = fc.get_stocks_from_list(stocks, ticker)

data = OrderedDict()

for ticker in tickers:
    data[ticker] = fc.get_time_series(ticker)

# converting dataframe data into panel
panel = pd.Panel(data)
panel.minor_axis = ['ticker',
                    'open',
                    'high',
                    'low',
                    'close',
                    'volume',
                    'ex-dividend',
                    'split_ratio',
                    'adj_open',
                    'adj_high',
                    'adj_low',
                    'adj_close',
                    'adj_volume']

panel.major_axis = panel.major_axis.tz_localize(pytz.utc)

buy_and_hold.stocks = tickers

# initializing trading enviroment
buy_and_hold_algo = TradingAlgorithm(initialize=buy_and_hold.initialize,
                                     handle_data=buy_and_hold.handle_data)

# run algo
perf_manual = buy_and_hold_algo.run(panel)

# calculation
total_pnl = perf_manual['pnl'][-1:].values

buy_trade = perf_manual[["status"]].loc[perf_manual['status'] == 'buy'].count()

sell_trade = perf_manual[["status"]].loc[perf_manual['status'] == 'sell'].count()

total_trade = buy_trade + sell_trade
