from collections import OrderedDict

import pandas as pd
from zipline.algorithm import TradingAlgorithm

import functions as fc
from strategies import buy_and_hold

tickers = ['AAPL', 'MSFT']

data = OrderedDict()

for ticker in tickers:
    data[ticker] = fc.get_time_series(ticker=ticker, start_date='2011-5-1', end_date='2014-1-5')
    # data[ticker].drop(['open', 'high', 'low', 'close', 'volume', 'ex-dividend', 'split_ratio'], axis=1)

# converting dataframe data into panel
panel = pd.Panel(data)

buy_and_hold.stocks = tickers

# initializing trading enviroment
buy_and_hold_algo = TradingAlgorithm(initialize=buy_and_hold.initialize,
                                     handle_data=buy_and_hold.handle_data)

# run algo
results = buy_and_hold_algo.run(panel)

# calculation
total_pnl = results['pnl'][-1:].values

buy_trade = results[["status"]].loc[results['status'] == 'buy'].count()

sell_trade = results[["status"]].loc[results['status'] == 'sell'].count()

total_trade = buy_trade + sell_trade
