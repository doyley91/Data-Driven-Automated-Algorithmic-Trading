"""
Module Docstring
"""

__author__ = "Gabriel Gauci Maistre"
__version__ = "0.1.0"
__license__ = "MIT"

import sys
from collections import OrderedDict, deque
from time import gmtime, strftime

import logbook as log
import numpy as np
import pandas as pd
import pyfolio as pf
import talib as ta
from sklearn.ensemble import RandomForestRegressor
from zipline.algorithm import TradingAlgorithm
from zipline.api import symbol, order_target_percent, record

import functions as fc


class MachineLearningRegressor(TradingAlgorithm):
    def initialize(self):
        """
        Called once at the start of the algorithm.
        The initialize function is the place to set your tradable universe and define any parameters
        """
        self.securities = tickers

        # Amount of prior bars to study
        self.window_length = 50

        # There needs to be enough data points to make a good model
        self.data_points = 100

        # Number of days to forecast
        self.pred_steps = 100

        # trading frequency, days
        self.trading_freq = 50

        # forecast increase to invest in
        self.forecast_difference = 10

        # Use a random forest regressor
        self.mdl = RandomForestRegressor()

        # Stores recent prices
        self.recent_prices = OrderedDict()

        self.invested = OrderedDict()

        for security in self.securities:
            self.recent_prices[security] = []
            self.invested[security] = False

        # Stores most recent prediction
        self.pred = deque(maxlen=self.pred_steps - 1)

        # schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())

    def before_trading_start(self, data):
        """
        Called every day before market open.
        """

    def handle_data(self, data):
        """
        Called every minute.
        """
        for security in self.securities:
            # Update the recent prices
            self.recent_prices[security].append(data.current(symbol(security), 'close'))

            # If there's enough recent price data
            if len(self.recent_prices[security]) >= self.window_length + 2:
                # Limit trading frequency
                # if len(self.recent_prices[security]) % self.trading_freq != 0.0:
                #   return

                # Stores the 15 and 50 day simple moving average
                sma15 = get_sma(close=self.recent_prices[security], days=15, window=self.window_length)
                sma50 = get_sma(close=self.recent_prices[security], days=50, window=self.window_length)

                # Independent, or input variables
                X = np.array(list(zip(sma15, sma50)))

                # Dependent, or output variable
                Y = self.recent_prices[security]

                if len(Y) >= self.data_points:
                    # Generate the model
                    self.mdl.fit(X, Y[self.window_length - 1:])

                    for k in range(1, self.pred_steps):
                        # Predict
                        self.pred.append(self.mdl.predict(X[-1:]))

                        sma15 = get_sma(close=np.append(self.recent_prices[security],
                                                        self.pred),
                                        days=15,
                                        window=self.window_length)

                        sma50 = get_sma(close=np.append(self.recent_prices[security],
                                                        self.pred),
                                        days=50,
                                        window=self.window_length)

                        X = np.array(list(zip(sma15, sma50)))

                    # If prediction goes up by a certain amount buy, else short
                    if (self.pred[-1] - self.pred[0]) > self.forecast_difference:
                        if not self.invested[security]:
                            order_target_percent(asset=symbol(security),
                                                 target=get_percentage_difference(first=self.pred[0],
                                                                                  last=self.pred[-1]))
                            self.invested[security] = True
                    elif (self.pred[-1] - self.pred[0]) < -self.forecast_difference:
                        if self.invested[security]:
                            order_target_percent(asset=symbol(security),
                                                 target=-get_percentage_difference(first=self.pred[0],
                                                                                   last=self.pred[-1]))
                            self.invested[security] = False

    def record_vars(self):
        """
        Plot variables at the end of each day.
        """
        record(prediction=int(self.pred[0]))


def get_sma(close, days, window):
    """
    Calculates the simple moving average of the security
    :param close: 
    :param days: 
    :param window: 
    :return: 
    """
    sma = ta.SMA(np.array(close), days)[window - 1:]

    # drop nan values
    sma = sma[~np.isnan(sma)]

    return sma


def get_percentage_difference(first, last):
    """
    Calculates the percentage of the portfolio to allocate based on the percentage increase
    :param first: 
    :param last: 
    :return: 
    """
    percent = ((last - first) / first) * 10

    percent = float(np.around(percent, 2))

    return percent


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
    returns, positions, transactions, gross_lev = pf.utils.extract_rets_pos_txn_from_zipline(results)

    """
    # plot of the top 5 drawdown periods
    pf.plot_drawdown_periods(returns, top=5).set_xlabel('Date')

    # create a full tear sheet for our algorithm. As an example, set the live start date to something arbitrary
    pf.create_full_tear_sheet(returns, positions=positions, transactions=transactions,
                              gross_lev=gross_lev, live_start_date='2009-10-22', round_trips=True)
    """
