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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from zipline.algorithm import TradingAlgorithm
from zipline.api import order_target_percent, symbol

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

        # initialise the model
        self.imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

    def handle_data(self, data):
        """
        Called every minute.
        """
        for security in self.securities:
            # Update the recent open prices
            self.recent_open_price[security].append(data.current(symbol(security), 'open'))

            # Update the recent prices
            self.recent_close_price[security].append(data.current(symbol(security), 'close'))

            if np.isnan(self.recent_open_price[security]).any():
                log.info('Warning: NaN found in {} open. Replacing with the mean value.'.format(security))
                # replace missing values
                self.recent_close_price[security] = fc.flatten_list(
                    self.imp.fit_transform(self.recent_close_price[security]).tolist())

            if np.isnan(self.recent_open_price[security]).any():
                log.info('Warning: NaN found in {} close. Replacing with the mean value.'.format(security))
                # replace missing values
                self.recent_open_price[security] = fc.flatten_list(
                    self.imp.fit_transform(self.recent_open_price[security]).tolist())

            # If there's enough recent price data
            if len(self.recent_close_price[security]) < self.window_length + 2:
                continue

            # Add independent variables, the prior changes
            sma2 = fc.get_sma(self.recent_close_price[security], 2, self.window_length)
            sma3 = fc.get_sma(self.recent_close_price[security], 3, self.window_length)
            sma4 = fc.get_sma(self.recent_close_price[security], 4, self.window_length)
            sma5 = fc.get_sma(self.recent_close_price[security], 5, self.window_length)
            sma6 = fc.get_sma(self.recent_close_price[security], 6, self.window_length)

            # make a list of 1's and 0's, 1 when the price increased from the prior bar
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
            #if len(self.recent_close_price[security]) % self.trading_freq != 0.0:
                #continue

            # there needs to be enough data points to make a good model
            if len(Y) <= self.data_points:
                continue

            # generate the model
            self.mdl.fit(X, Y)

            # predict tomorrow's movement
            pred = self.mdl.predict(X[-1:])

            # the amount to allocate per security
            allocation = 1 / len(self.securities)

            # if prediction = 1
            if not pred:
                continue

            # check if we don't currently hold a position
            if self.portfolio.positions[symbol(security)].amount:
                continue

            order_target_percent(asset=symbol(security), target=allocation)
            order_target_percent(asset=symbol(security), target=-allocation,
                                 stop_price=self.recent_close_price[security][-1] * 0.80)

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

    print("Machine Learning Classifier strategy results\n"
          "-------------\n"
          "Total capital used: {:.2f}\n"
          "Sharpe Ratio: {:.3f}\n"
          "Portfolio Value: {:.3f}\n"
          "Algorithm Period Return: {:.3f}\n"
          "Benchmark Period Return: {:.3f}\n"
          "Algorithm Volatility: {:.3f}\n"
          "Benchmark Volatility: {:.3f}".format(-results.capital_used.sum(),
                                                results.sharpe[-1],
                                                results.portfolio_value[-1],
                                                results.algorithm_period_return[-1],
                                                results.benchmark_period_return[-1],
                                                results.algo_volatility[-1],
                                                results.benchmark_volatility[-1]))

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
    fig.show()
    fig.savefig('charts/MLC-Portfolio-Benchmark-{}.png'.format(strftime("%Y-%m-%d-%H:%M:%S", gmtime())))
