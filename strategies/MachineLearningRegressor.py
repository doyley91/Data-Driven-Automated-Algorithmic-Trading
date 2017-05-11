import functions as fc
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from collections import OrderedDict
from zipline.algorithm import TradingAlgorithm
from zipline.api import symbol, order, record
import talib as ta


class MachineLearningRegressor(TradingAlgorithm):
    def initialize(context):
        """
        Called once at the start of the algorithm.
        """
        context.window_length = 50  # Amount of prior bars to study

        context.mdl = RandomForestRegressor()  # Use a random forest classifier

        # deques are lists with a maximum length where old entries are shifted out
        context.recent_prices = OrderedDict()  # Stores recent prices

        for ticker in tickers:
            context.recent_prices[ticker] = []

        context.sma15 = []
        context.sma50 = []

        context.X = []  # Independent, or input variables
        context.Y = []  # Dependent, or output variable

        context.pred = 0  # Stores most recent prediction

    def handle_data(context, data):
        """
        Called every minute.
        """
        for ticker in tickers:
            context.recent_prices[ticker].append(data.current(symbol(ticker), 'close'))  # Update the recent prices
            if len(context.recent_prices[ticker]) >= context.window_length + 2:  # If there's enough recent price data
                # Add independent variables, the prior changes
                context.sma15 = ta.SMA(np.array(context.recent_prices[ticker]), 15)[context.window_length-1:]
                # drop nan values
                context.sma15 = context.sma15[~np.isnan(context.sma15)]

                # Add independent variables, the prior changes
                context.sma50 = ta.SMA(np.array(context.recent_prices[ticker]), 50)
                # drop nan values
                context.sma50 = context.sma50[~np.isnan(context.sma50)]

                context.X = np.array(list(zip(context.sma15, context.sma50)))
                context.Y = context.recent_prices[ticker]  # Add dependent variable, the final change

                if len(context.Y) >= 100:  # There needs to be enough data points to make a good model
                    context.mdl.fit(context.X, context.Y[context.window_length-1:])  # Generate the model

                    context.pred = context.mdl.predict(context.X[-1:])  # Predict

                    # If prediction = 1, buy all shares affordable, if 0 sell all shares
                    # order(asset=symbol(ticker), amount=100)
                    order(asset=symbol(ticker), amount=100)

                    record(prediction=int(context.pred))

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT']

    data = OrderedDict()

    for ticker in tickers:
        data[ticker] = fc.get_time_series(ticker=ticker,
                                          start_date='2010-1-1',
                                          end_date='2017-1-1')

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
    Strategy = MachineLearningRegressor()
    # #print df

    # # # # # # Run Strategy
    results = Strategy.run(panel)
    results['algorithm_returns'] = (1 + results.returns).cumprod()

    results.to_csv('output.csv')
    print(results['algorithm_returns'].tail(1)[0] * 100)
