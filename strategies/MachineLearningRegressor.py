# Use a random forest classifier. More here: http://scikit-learn.org/stable/user_guide.html
import functions as fc
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from collections import OrderedDict, deque
import numpy as np
from zipline.algorithm import TradingAlgorithm
from zipline.api import record, order, symbol, order_target_percent


class MachineLearningClassifier(TradingAlgorithm):
    def initialize(context):
        context.window_length = 3  # Amount of prior bars to study

        context.mdl = RandomForestRegressor()  # Use a random forest classifier

        # deques are lists with a maximum length where old entries are shifted out
        context.recent_prices = deque(maxlen=400)  # Stores recent prices
        context.X = deque(maxlen=500)  # Independent, or input variables
        context.Y = deque(maxlen=500)  # Dependent, or output variable

        context.pred = 0  # Stores most recent prediction

    def handle_data(context, data):
        for ticker in tickers:
            context.recent_prices.append(data[symbol(ticker)]['close'])  # Update the recent prices
            if len(context.recent_prices) == context.window_length + 2:  # If there's enough recent price data

                # Make a list of 1's and 0's, 1 when the price increased from the prior bar
                changes = np.diff(context.recent_prices) > 0

                context.X.append(changes[:-1])  # Add independent variables, the prior changes
                context.Y.append(changes[-1])  # Add dependent variable, the final change

                if len(context.Y) >= 100:  # There needs to be enough data points to make a good model
                    context.mdl.fit(context.X, context.Y)  # Generate the model

                    context.pred = context.mdl.predict(changes[1:])  # Predict

                    # If prediction = 1, buy all shares affordable, if 0 sell all shares
                    # order(asset=symbol(ticker), amount=100)
                    order_target_percent(asset=symbol(ticker), target=context.pred)

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
