"""
Module Docstring
"""

__author__ = "Gabriel Gauci Maistre"
__version__ = "0.1.0"
__license__ = "MIT"

import sys
from collections import OrderedDict
from time import gmtime, strftime

import logbook as log
import numpy as np
import pandas as pd
import pyfolio as pf
import talib as ta
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from zipline.algorithm import TradingAlgorithm
from zipline.api import record, order_target_percent, symbol

import functions as fc


class MachineLearningClassifier(TradingAlgorithm):
    def initialize(self):
        """
        Called once at the start of the algorithm.
        """
        self.securities = tickers

        # Amount of prior bars to study
        self.window_length = 6

        self.data_points = 100

        # trading frequency, days
        self.trading_freq = 20

        # Use a random forest classifier
        self.mdl = RandomForestClassifier()

        # Stores recent open prices
        self.recent_open_price = OrderedDict()

        # Stores recent close prices
        self.recent_close_price = OrderedDict()

        self.invested = OrderedDict()
        self.sma2_result = OrderedDict()
        self.sma3_result = OrderedDict()
        self.sma4_result = OrderedDict()
        self.sma5_result = OrderedDict()
        self.sma6_result = OrderedDict()
        self.result = OrderedDict()

        for security in self.securities:
            self.recent_open_price[security] = []
            self.recent_close_price[security] = []
            self.sma2_result[security] = []
            self.sma3_result[security] = []
            self.sma4_result[security] = []
            self.sma5_result[security] = []
            self.sma6_result[security] = []
            self.result[security] = []
            self.invested[security] = False

    def handle_data(self, data):
        """
        Called every minute.
        """
        for security in self.securities:
            # Update the recent open prices
            self.recent_open_price[security].append(data.current(symbol(security), 'open'))

            # Update the recent prices
            self.recent_close_price[security].append(data.current(symbol(security), 'close'))

            # If there's enough recent price data
            if len(self.recent_close_price[security]) < self.window_length + 2:
                return

            # Add independent variables, the prior changes
            sma2 = get_sma(self.recent_close_price[security], 2, self.window_length)
            sma3 = get_sma(self.recent_close_price[security], 3, self.window_length)
            sma4 = get_sma(self.recent_close_price[security], 4, self.window_length)
            sma5 = get_sma(self.recent_close_price[security], 5, self.window_length)
            sma6 = get_sma(self.recent_close_price[security], 6, self.window_length)

            # Make a list of 1's and 0's, 1 when the price increased from the prior bar
            self.sma2_result[security] = np.append(self.sma2_result[security],
                                                   self.recent_close_price[security][-1:] > sma2[-1:])
            self.sma3_result[security] = np.append(self.sma3_result[security],
                                                   self.recent_close_price[security][-1:] > sma3[-1:])

            self.sma4_result[security] = np.append(self.sma4_result[security],
                                                   self.recent_close_price[security][-1:] > sma4[-1:])

            self.sma5_result[security] = np.append(self.sma5_result[security],
                                                   self.recent_close_price[security][-1:] > sma5[-1:])

            self.sma6_result[security] = np.append(self.sma6_result[security],
                                                   self.recent_close_price[security][-1:] > sma6[-1:])

            self.result[security] = np.append(self.result[security],
                                              self.recent_close_price[security][-1:] >
                                              self.recent_open_price[security][-1:])

            # Add independent variables, the prior changes
            X = np.array(list(zip(self.sma2_result[security],
                                  self.sma3_result[security],
                                  self.sma4_result[security],
                                  self.sma5_result[security],
                                  self.sma6_result[security])))

            # Add dependent variable, the final change
            Y = self.result[security]

            # limit trading frequency
            if len(self.recent_close_price[security]) % self.trading_freq != 0.0:
                continue

            # there needs to be enough data points to make a good model
            if len(Y) >= self.data_points:
                # generate the model
                self.mdl.fit(X, Y)

                # predict tomorrow's movement
                pred = self.mdl.predict(X[-1:])

                # the amount to allocate per security
                allocation = 1 / len(self.securities)

                # if prediction = 1
                if pred:
                    # check if we don't currently hold a position
                    if not self.invested[security]:
                        order_target_percent(asset=symbol(security), target=allocation)
                        self.invested[security] = True
                # if prediction = 0
                else:
                    # check if we currently hold a position
                    if self.invested[security]:
                        order_target_percent(asset=symbol(security), target=-allocation)
                        self.invested[security] = False

                # plot variables at the end of each day
                record(prediction=int(pred))


def get_sma(close, days, window):
    """
    calculates the simple moving average
    :param close: 
    :param days: 
    :param window: 
    :return: 
    """
    sma = ta.SMA(np.array(close), days)[window - 1:]

    # drop nan values
    sma = sma[~np.isnan(sma)]

    return sma


if __name__ == '__main__':
    """ 
    This is executed when run from the command line 
    """
    # enable zipline debug log
    zipline_logging = log.NestedSetup([
        log.NullHandler(level=log.DEBUG),
        log.StreamHandler(sys.stdout, level=log.INFO),
        log.StreamHandler(sys.stderr, level=log.ERROR),
    ])
    zipline_logging.push_application()

    start = '2010-1-1'

    end = '2017-1-1'

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
    Strategy = MachineLearningClassifier()

    # run strategy
    results = Strategy.run(panel)

    # calculate cumulative returns of the algorithm
    results['algorithm_returns'] = (1 + results.returns).cumprod()

    # save the results to a csv
    results.to_csv('results/mlc-results-{}.csv'.format(strftime("%Y-%m-%d-%H:%M:%S", gmtime())))

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

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(data[benchmark]['close'])
    ax2.plot(results.portfolio_value)
    ax1.set(title='Benchmark', xlabel='time', ylabel='$')
    ax2.set(title='Portfolio', xlabel='time', ylabel='$')
    ax1.legend(['$'])
    ax2.legend(['$'])
    fig.tight_layout()
    fig.savefig('charts/Portfolio-Benchmark-{}.png'.format(strftime("%Y-%m-%d-%H:%M:%S", gmtime())))
