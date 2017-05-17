# Use a random forest classifier. More here: http://scikit-learn.org/stable/user_guide.html
from collections import OrderedDict

import numpy as np
import pandas as pd
import talib as ta
from sklearn.ensemble import RandomForestClassifier
from zipline.algorithm import TradingAlgorithm
from zipline.api import record, order, symbol

import functions as fc


class MachineLearningClassifier(TradingAlgorithm):
    def initialize(context):
        """
        Called once at the start of the algorithm.
        """
        context.window_length = 6  # Amount of prior bars to study

        context.data_points = 100

        context.forecast_steps = 100  # Number of days to forecast

        context.forecast = []

        context.mdl = RandomForestClassifier()  # Use a random forest classifier

        context.sma2 = context.sma3 = context.sma4 = context.sma5 = context.sma6 = []

        context.sma2_result = context.sma3_result = context.sma4_result = context.sma5_result = context.sma6_result = []

        # deques are lists with a maximum length where old entries are shifted out
        context.recent_open_price = OrderedDict()  # Stores recent open prices
        context.recent_close_price = OrderedDict()  # Stores recent close prices
        context.sma_2_result = OrderedDict()
        context.sma_3_result = OrderedDict()
        context.sma_4_result = OrderedDict()
        context.sma_5_result = OrderedDict()
        context.sma_6_result = OrderedDict()
        context.result = OrderedDict()

        for ticker in tickers:
            context.recent_open_price[ticker] = []
            context.recent_close_price[ticker] = []
            context.sma_2_result[ticker] = []
            context.sma_3_result[ticker] = []
            context.sma_4_result[ticker] = []
            context.sma_5_result[ticker] = []
            context.sma_6_result[ticker] = []
            context.result[ticker] = []

        context.X = []  # Independent, or input variables
        context.Y = []  # Dependent, or output variable

        context.pred = 0  # Stores most recent prediction

    def handle_data(context, data):
        """
        Called every minute.
        """
        for ticker in tickers:
            context.recent_open_price[ticker].append(data.current(symbol(ticker), 'open'))  # Update the recent prices
            context.recent_close_price[ticker].append(data.current(symbol(ticker), 'close'))  # Update the recent prices

            # If there's enough recent price data
            if len(context.recent_close_price[ticker]) >= context.window_length + 2:
                # Add independent variables, the prior changes
                context.sma2 = get_sma(context.recent_close_price[ticker], 2, context.window_length)
                context.sma3 = get_sma(context.recent_close_price[ticker], 3, context.window_length)
                context.sma4 = get_sma(context.recent_close_price[ticker], 4, context.window_length)
                context.sma5 = get_sma(context.recent_close_price[ticker], 5, context.window_length)
                context.sma6 = get_sma(context.recent_close_price[ticker], 6, context.window_length)

                # Make a list of 1's and 0's, 1 when the price increased from the prior bar
                context.sma_2_result[ticker] = np.append(context.sma_2_result[ticker],
                                                         is_x_higher_than_y(context.recent_close_price[ticker][-1:],
                                                                            context.sma2[-1:]))
                context.sma_3_result[ticker] = np.append(context.sma_3_result[ticker],
                                                         is_x_higher_than_y(context.recent_close_price[ticker][-1:],
                                                                            context.sma3[-1:]))

                context.sma_4_result[ticker] = np.append(context.sma_4_result[ticker],
                                                         is_x_higher_than_y(context.recent_close_price[ticker][-1:],
                                                                            context.sma4[-1:]))

                context.sma_5_result[ticker] = np.append(context.sma_5_result[ticker],
                                                         is_x_higher_than_y(context.recent_close_price[ticker][-1:],
                                                                            context.sma5[-1:]))

                context.sma_6_result[ticker] = np.append(context.sma_6_result[ticker],
                                                         is_x_higher_than_y(context.recent_close_price[ticker][-1:],
                                                                            context.sma6[-1:]))

                context.result[ticker] = np.append(context.result[ticker],
                                                   is_x_higher_than_y(context.recent_close_price[ticker][-1:],
                                                                      context.recent_open_price[ticker][-1:]))

                # Add independent variables, the prior changes
                context.X = np.array(list(zip(context.sma_2_result[ticker],
                                              context.sma_3_result[ticker],
                                              context.sma_4_result[ticker],
                                              context.sma_5_result[ticker],
                                              context.sma_6_result[ticker])))

                context.Y = context.result[ticker]  # Add dependent variable, the final change

                if len(context.Y) >= context.data_points:  # There needs to be enough data points to make a good model
                    context.mdl.fit(context.X, context.Y)  # Generate the model

                    context.pred = context.mdl.predict(context.X[-1:])  # Predict

                    for k in range(1, context.forecast_steps):
                        context.forecast = np.append(context.forecast, context.pred)

                        sma2 = ta.SMA(np.array(context.recent_close_price[ticker]), 2)[context.window_length - 1:]
                        sma3 = ta.SMA(np.array(context.recent_close_price[ticker]), 3)[context.window_length - 1:]
                        sma4 = ta.SMA(np.array(context.recent_close_price[ticker]), 4)[context.window_length - 1:]
                        sma5 = ta.SMA(np.array(context.recent_close_price[ticker]), 5)[context.window_length - 1:]
                        sma6 = ta.SMA(np.array(context.recent_close_price[ticker]), 6)[context.window_length - 1:]

                        context.sma2_result = np.append(context.sma2_result,
                                                        is_x_higher_than_y(context.forecast[-1:],
                                                                           sma2[-1:]))
                        context.sma3_result = np.append(context.sma3_result,
                                                        is_x_higher_than_y(context.forecast[-1:],
                                                                           sma3[-1:]))
                        context.sma4_result = np.append(context.sma4_result,
                                                        is_x_higher_than_y(context.forecast[-1:],
                                                                           sma4[-1:]))
                        context.sma5_result = np.append(context.sma5_result,
                                                        is_x_higher_than_y(context.forecast[-1:],
                                                                           sma5[-1:]))
                        context.sma6_result = np.append(context.sma6_result,
                                                        is_x_higher_than_y(context.forecast[-1:],
                                                                           sma6[-1:]))

                        context.X = np.array(list(zip(context.sma2_result,
                                                      context.sma3_result,
                                                      context.sma4_result,
                                                      context.sma5_result,
                                                      context.sma6_result)))

                        context.pred = context.mdl.predict(context.X[-1:])

                    # If prediction = 1, buy all shares affordable, if 0 sell all shares
                    order(asset=symbol(ticker), amount=100)

                    record(prediction=int(context.pred))


def is_x_higher_than_y(x, y):
    return 1 if x > y else 0


def get_sma(close, days, window):
    sma = ta.SMA(np.array(close), days)[window - 1:]

    # drop nan values
    sma = sma[~np.isnan(sma)]

    return sma


if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT']

    data = OrderedDict()

    for ticker in tickers:
        data[ticker] = fc.get_time_series(ticker=ticker,
                                          start_date='2011-5-1',
                                          end_date='2014-1-5')

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

    results.to_csv('output.csv')
    print(results['algorithm_returns'].tail(1)[0] * 100)
