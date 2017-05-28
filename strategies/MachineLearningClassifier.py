import sys
from collections import OrderedDict

import logbook as log
import numpy as np
import pandas as pd
import pyfolio as pf
import talib as ta
from sklearn.ensemble import RandomForestClassifier
from zipline.algorithm import TradingAlgorithm
from zipline.api import record, order_target_percent, symbol, get_datetime

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

        self.day_count = -1

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

        # Independent, or input variables
        self.X = []

        # Dependent, or output variable
        self.Y = []

        # Stores most recent prediction
        self.pred = 0

    def handle_data(self, data):
        """
        Called every minute.
        """
        for security in self.securities:
            self.recent_open_price[security].append(data.current(symbol(security), 'open'))  # Update the recent prices
            self.recent_close_price[security].append(
                data.current(symbol(security), 'close'))  # Update the recent prices

            # If there's enough recent price data
            if len(self.recent_close_price[security]) >= self.window_length + 2:
                # Trade only once per day
                loc_dt = pd.Timestamp(get_datetime()).tz_convert('US/Eastern')
                if loc_dt.hour == 16 and loc_dt.minute == 0:
                    self.day_count += 1
                    pass
                else:
                    return

                # Limit trading frequency
                if len(self.recent_close_price[security]) % self.trading_freq != 0.0:
                    return

                # Add independent variables, the prior changes
                self.sma2 = get_sma(self.recent_close_price[security], 2, self.window_length)
                self.sma3 = get_sma(self.recent_close_price[security], 3, self.window_length)
                self.sma4 = get_sma(self.recent_close_price[security], 4, self.window_length)
                self.sma5 = get_sma(self.recent_close_price[security], 5, self.window_length)
                self.sma6 = get_sma(self.recent_close_price[security], 6, self.window_length)

                # Make a list of 1's and 0's, 1 when the price increased from the prior bar
                self.sma2_result[security] = np.append(self.sma2_result[security],
                                                       is_x_higher_than_y(self.recent_close_price[security][-1:],
                                                                          self.sma2[-1:]))
                self.sma3_result[security] = np.append(self.sma3_result[security],
                                                       is_x_higher_than_y(self.recent_close_price[security][-1:],
                                                                          self.sma3[-1:]))

                self.sma4_result[security] = np.append(self.sma4_result[security],
                                                       is_x_higher_than_y(self.recent_close_price[security][-1:],
                                                                          self.sma4[-1:]))

                self.sma5_result[security] = np.append(self.sma5_result[security],
                                                       is_x_higher_than_y(self.recent_close_price[security][-1:],
                                                                          self.sma5[-1:]))

                self.sma6_result[security] = np.append(self.sma6_result[security],
                                                       is_x_higher_than_y(self.recent_close_price[security][-1:],
                                                                          self.sma6[-1:]))

                self.result[security] = np.append(self.result[security],
                                                  is_x_higher_than_y(self.recent_close_price[security][-1:],
                                                                     self.recent_open_price[security][-1:]))

                # Add independent variables, the prior changes
                self.X = np.array(list(zip(self.sma2_result[security],
                                           self.sma3_result[security],
                                           self.sma4_result[security],
                                           self.sma5_result[security],
                                           self.sma6_result[security])))

                # Add dependent variable, the final change
                self.Y = self.result[security]

                # There needs to be enough data points to make a good model
                if len(self.Y) >= self.data_points:
                    # Generate the model
                    self.mdl.fit(self.X, self.Y)

                    # Predict
                    self.pred = self.mdl.predict(self.X[-1:])

                    # If prediction = 1, buy all shares affordable, if 0 sell all shares
                    if self.pred:
                        if not self.invested[security]:
                            order_target_percent(asset=symbol(security), target=1.0)
                            self.invested[security] = True
                    else:
                        if self.invested[security]:
                            order_target_percent(asset=symbol(security), target=-1.0)
                            self.invested[security] = False

    def record_vars(self):
        """
        Plot variables at the end of each day.
        """
        record(prediction=int(self.pred))


def is_x_higher_than_y(x, y):
    """
    1 if forecast price is higher than today's price, else 0
    :param x: 
    :param y: 
    :return: 
    """
    return 1 if x > y else 0


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
    zipline_logging = log.NestedSetup([
        log.NullHandler(level=log.DEBUG),
        log.StreamHandler(sys.stdout, level=log.INFO),
        log.StreamHandler(sys.stderr, level=log.ERROR),
    ])
    zipline_logging.push_application()

    start = '2010-1-1'

    end = '2017-1-1'

    tickers = ['MSFT', 'CDE', 'NAVB', 'HRG', 'HL']

    benchmark = 'GSPC'

    data = OrderedDict()

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

    # converting dataframe data into panel
    panel = pd.Panel(data)

    # # # # init Strat Class
    Strategy = MachineLearningClassifier()
    # #print df

    # # # # # # Run Strategy
    results = Strategy.run(panel)
    results['algorithm_returns'] = (1 + results.returns).cumprod()

    results.to_csv('data/mlc-results.csv')

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
