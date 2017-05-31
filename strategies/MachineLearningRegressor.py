"""
Module Docstring
"""

__author__ = "Gabriel Gauci Maistre"
__version__ = "0.1.0"
__license__ = "MIT"

import sys
from collections import OrderedDict
from time import gmtime, strftime

import logbook
import numpy as np
import pandas as pd
import pyfolio as pf
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from zipline.algorithm import TradingAlgorithm
from zipline.api import symbol, order_target_percent, record, get_datetime

import functions as fc


class MachineLearningRegressor(TradingAlgorithm):
    def initialize(self):
        """
        Called once at the start of the algorithm.
        The initialize function is the place to set your tradable universe and define any parameters
        """
        self.securities = tickers

        # there needs to be enough data points to make a good model
        self.data_points = 100

        # amount of prior bars to study
        self.window_length = 50

        # trading frequency, days
        self.trading_freq = 20

        # Use a random forest regressor
        self.mdl = RandomForestRegressor()

        # stores recent prices
        self.recent_prices = OrderedDict()

        # whether we currently hold a position in the stock or not
        self.invested = OrderedDict()

        for security in self.securities:
            self.recent_prices[security] = []
            self.invested[security] = False

        # initialise the model
        self.imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

    def handle_data(self, data):
        """
        Called every minute.
        """
        for security in self.securities:
            # update the recent prices
            self.recent_prices[security].append(data.current(symbol(security), 'close'))

            if np.isnan(self.recent_prices[security]).any():
                print('Warning: NaN found in', security, 'close at {}. Replacing with the mean value.'.format(
                    pd.Timestamp(get_datetime()).tz_convert('US/Eastern')))
                # replace missing values
                self.recent_prices[security] = fc.flatten_list(
                    self.imp.fit_transform(self.recent_prices[security]).tolist())

            # there needs to be enough data points to make a good model
            if len(self.recent_prices[security]) <= self.data_points:
                continue

            # limit trading frequency
            if len(self.recent_prices[security]) % self.trading_freq != 0.0:
                continue

            # stores the 15 and 50 day simple moving average
            sma15 = fc.get_sma(close=self.recent_prices[security], days=15, window=self.window_length)
            sma50 = fc.get_sma(close=self.recent_prices[security], days=50, window=self.window_length)

            # independent, or input variables
            X = np.array(list(zip(sma15, sma50)))

            # dependant, or output variables
            Y = self.recent_prices[security]

            # generate the model
            self.mdl.fit(X, Y[self.window_length - 1:])

            # predict tomorrow's close
            pred = self.mdl.predict(X[-1:])

            # the amount to allocate per security
            allocation = 1 / len(self.securities)

            # buy if predicted price goes up
            if pred > self.recent_prices[security][-1:]:
                # check if we don't currently hold a position
                if not self.invested[security]:
                    order_target_percent(asset=symbol(security), target=allocation)
                    self.invested[security] = True
            # short if if predicted price goes down
            else:
                # check if we currently hold a position
                if self.invested[security]:
                    order_target_percent(asset=symbol(security), target=-allocation)
                    self.invested[security] = False


if __name__ == '__main__':
    """ 
    This is executed when run from the command line 
    """
    # enable zipline debug log
    log_format = "{record.extra[algo_dt]}  {record.message}"

    zipline_logging = logbook.NestedSetup([
        logbook.NullHandler(level=logbook.DEBUG),
        logbook.StreamHandler(sys.stdout, level=logbook.INFO, format_string=log_format),
        logbook.StreamHandler(sys.stdout, level=logbook.DEBUG, format_string=log_format),
        logbook.StreamHandler(sys.stdout, level=logbook.WARNING, format_string=log_format),
        logbook.StreamHandler(sys.stdout, level=logbook.NOTICE, format_string=log_format),
        logbook.StreamHandler(sys.stderr, level=logbook.ERROR, format_string=log_format),
    ])
    zipline_logging.push_application()

    log = logbook.Logger('Main Logger')

    start = '2010-1-1'

    end = '2017-1-1'

    # tickers to pass to the algorithm
    tickers = ['MSFT', 'CDE', 'NAVB', 'HRG', 'HL']

    # index to benchmark the algorithm
    benchmark = 'GSPC'

    # initialising an ordered dictionary to store all our stocks
    data = OrderedDict()

    # tidying the data for the backtester
    for ticker in tickers:
        data[ticker] = fc.get_time_series(ticker=ticker,
                                          start_date=start,
                                          end_date=end)

        data[ticker].drop(['open',
                           'high',
                           'low',
                           'close',
                           'ex-dividend',
                           'split_ratio'],
                          axis=1,
                          inplace=True)

        data[ticker].rename(columns={'ticker': 'sid',
                                     'adj_open': 'open',
                                     'adj_high': 'high',
                                     'adj_low': 'low',
                                     'adj_close': 'close'},
                            inplace=True)

    # converting data frame data into panel
    panel = pd.Panel(data)

    # initialise strategy class
    algo = MachineLearningRegressor()

    # run strategy
    results = algo.run(panel)

    # calculate cumulative returns of the algorithm
    results['algorithm_returns'] = (1 + results.returns).cumprod()

    # save the results to a csv
    results.to_csv('results/mlr-results-{}.csv'.format(strftime("%Y-%m-%d-%H:%M:%S", gmtime())))

    data[benchmark] = fc.get_time_series(ticker=benchmark,
                                         start_date=start,
                                         end_date=end,
                                         file_location='data/GSPC.csv')

    data[benchmark].drop(['close'],
                         axis=1,
                         inplace=True)

    data[benchmark].rename(columns={'ticker': 'sid',
                                    'adj_close': 'close'},
                           inplace=True)

    # get the returns, positions, and transactions from the zipline backtest object
    returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)

    # plot the portfolio value against the benchmark
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(data[benchmark]['close'])
    ax2.plot(results.portfolio_value)
    ax1.set(title='Benchmark', xlabel='time', ylabel='$')
    ax2.set(title='Portfolio', xlabel='time', ylabel='$')
    ax1.legend(['^GSPC'])
    ax2.legend(['Portfolio'])
    fig.tight_layout()
    fig.savefig('charts/MLR-Portfolio-Benchmark-{}.png'.format(strftime("%Y-%m-%d-%H:%M:%S", gmtime())))
