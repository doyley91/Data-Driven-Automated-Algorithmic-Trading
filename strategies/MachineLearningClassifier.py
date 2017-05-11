# Use a random forest classifier. More here: http://scikit-learn.org/stable/user_guide.html
import functions as fc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
import numpy as np
from zipline.algorithm import TradingAlgorithm
from zipline.api import record, order, symbol
import talib as ta


class MachineLearningClassifier(TradingAlgorithm):
    def initialize(context):
        """
        Called once at the start of the algorithm.
        """
        context.window_length = 6  # Amount of prior bars to study

        context.mdl = RandomForestClassifier()  # Use a random forest classifier

        # deques are lists with a maximum length where old entries are shifted out
        context.recent_open_price = OrderedDict()  # Stores recent open prices
        context.recent_close_price = OrderedDict()  # Stores recent close prices
        context.sma2 = OrderedDict()
        context.sma3 = OrderedDict()
        context.sma4 = OrderedDict()
        context.sma5 = OrderedDict()
        context.sma6 = OrderedDict()
        context.sma6 = OrderedDict()
        context.result = OrderedDict()

        for ticker in tickers:
            context.recent_open_price[ticker] = []
            context.recent_close_price[ticker] = []
            context.sma2[ticker] = []
            context.sma3[ticker] = []
            context.sma4[ticker] = []
            context.sma5[ticker] = []
            context.sma6[ticker] = []
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
            if len(context.recent_close_price[ticker]) == context.window_length + 2:  # If there's enough recent price data
                # Add independent variables, the prior changes
                context.sma2[ticker] = ta.SMA(np.array(context.recent_close_price[ticker]), 2)[context.window_length - 1:]
                # drop nan values
                context.sma2[ticker] = context.sma2[ticker][~np.isnan(context.sma2[ticker])]

                # Add independent variables, the prior changes
                context.sma3[ticker] = ta.SMA(np.array(context.recent_close_price[ticker]), 3)
                # drop nan values
                context.sma3[ticker] = context.sma3[ticker][~np.isnan(context.sma3[ticker])]

                # Add independent variables, the prior changes
                context.sma4[ticker] = ta.SMA(np.array(context.recent_close_price[ticker]), 4)[context.window_length - 1:]
                # drop nan values
                context.sma4[ticker] = context.sma4[ticker][~np.isnan(context.sma4[ticker])]

                # Add independent variables, the prior changes
                context.sma5[ticker] = ta.SMA(np.array(context.recent_close_price[ticker]), 5)[context.window_length - 1:]
                # drop nan values
                context.sma5[ticker] = context.sma5[ticker][~np.isnan(context.sma5[ticker])]

                # Add independent variables, the prior changes
                context.sma6[ticker] = ta.SMA(np.array(context.recent_close_price[ticker]), 6)
                # drop nan values
                context.sma6[ticker] = context.sma6[ticker][~np.isnan(context.sma6[ticker])]

                # Make a list of 1's and 0's, 1 when the price increased from the prior bar
                context.sma2 = np.apply_along_axis(
                    func1d=lambda x: 1 if context.recent_close_price > context.sma2[ticker] else 0,
                    axis=1,
                    arr=context.sma2[ticker])

                context.sma3 = np.apply_along_axis(
                    func1d=lambda x: 1 if context.recent_close_price > context.sma3[ticker] else 0,
                    axis=1,
                    arr=context.sma3[ticker])

                context.sma4 = np.apply_along_axis(
                    func1d=lambda x: 1 if context.recent_close_price > context.sma4[ticker] else 0,
                    axis=1,
                    arr=context.sma4[ticker])

                context.sma5 = np.apply_along_axis(
                    func1d=lambda x: 1 if context.recent_close_price > context.sma5[ticker] else 0,
                    axis=1,
                    arr=context.sma5[ticker])

                context.sma6 = np.apply_along_axis(
                    func1d=lambda x: 1 if context.recent_close_price > context.sma6[ticker] else 0,
                    axis=1,
                    arr=context.sma6[ticker])

                context.result[ticker] = np.apply_along_axis(
                    func1d=lambda x: 1 if context.recent_close_price > context.recent_open_price[ticker] else 0,
                    axis=1,
                    arr=context.result[ticker])

                # Add independent variables, the prior changes
                context.X = np.array(list(zip(context.sma2, context.sma3, context.sma4, context.sma5, context.sma6)))
                context.Y = context.result  # Add dependent variable, the final change

                if len(context.Y) >= 100:  # There needs to be enough data points to make a good model
                    context.mdl.fit(context.X, context.Y)  # Generate the model

                    context.pred = context.mdl.predict(context.X[-1:])  # Predict

                    # If prediction = 1, buy all shares affordable, if 0 sell all shares
                    # order(asset=symbol(ticker), amount=100)
                    order(asset=symbol(ticker), ammount=100)

                    record(prediction=int(context.pred))


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
