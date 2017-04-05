import functions as fc
import pytz
import pandas as pd
from zipline.algorithm import TradingAlgorithm
from strategies import buy, buy_and_hold, dual_ema
from collections import OrderedDict

stocks = ['AAPL', 'MSFT']

data = OrderedDict()

for stock in stocks:
    data[stock] = fc.get_time_series(stock)

#converting dataframe data into panel
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

buy.stocks = buy_and_hold.stocks = dual_ema.stocks = stocks

#initializing trading enviroment
buy_algo = TradingAlgorithm(initialize=buy.initialize,
                            handle_data=buy.handle_data,
                            capital_base=100000.0)

buy_and_hold_algo = TradingAlgorithm(initialize=buy_and_hold.initialize,
                                     handle_data=buy_and_hold.handle_data,
                                     capital_base=100000.0)

dual_ema_algo = TradingAlgorithm(initialize=dual_ema.initialize,
                                 handle_data=dual_ema.handle_data,
                                 capital_base=100000.0)

#run algo
perf_manual = buy_algo.run(panel)
perf_manual = buy_and_hold_algo.run(panel)
perf_manual = dual_ema_algo.run(panel)

#calculation
perf_manual['pnl'][-1:].values

buy_trade = perf_manual[["status"]].loc[perf_manual["status"] == "buy"].count()

sell_trade = perf_manual[["status"]].loc[perf_manual["status"] == "sell"].count()

total_trade = buy_trade + sell_trade
