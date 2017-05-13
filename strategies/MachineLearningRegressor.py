import functions as fc
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from collections import OrderedDict
from zipline.algorithm import TradingAlgorithm
from zipline.api import symbol, order_target_percent, record, set_commission, set_slippage
from zipline.finance import commission, slippage
import talib as ta


class MachineLearningRegressor(TradingAlgorithm):
    def initialize(context):
        """
        Called once at the start of the algorithm.
        """
        set_slippage(slippage.FixedSlippage(spread=0.00))
        set_commission(commission.PerShare(cost=0, min_trade_cost=0))

        context.invested = False

        context.securities = tickers

        context.window_length = 50  # Amount of prior bars to study

        context.data_points = 100  # There needs to be enough data points to make a good model

        context.forecast_steps = 100  # Number of days to forecast

        context.trading_freq = 50  # trading frequency, days

        context.forecast_difference = 15

        context.forecast = []

        context.mdl = RandomForestRegressor()  # Use a random forest classifier

        # deques are lists with a maximum length where old entries are shifted out
        context.recent_prices = OrderedDict()  # Stores recent prices

        for ticker in tickers:
            context.recent_prices[ticker] = []

        context.sma15 = context.sma50 = []

        context.X = []  # Independent, or input variables
        context.Y = []  # Dependent, or output variable

        context.pred = []  # Stores most recent prediction

    def before_trading_start(context, data):
        """
        Called every day before market open.
        """

    def handle_data(context, data):
        """
        Called every minute.
        """
        for security in context.securities:
            context.recent_prices[security].append(data.current(symbol(security), 'close'))  # Update the recent prices
            if len(context.recent_prices[security]) >= context.window_length + 2:  # If there's enough recent price data
                # Limit trading frequency
                if len(context.recent_prices[security]) % context.trading_freq != 0.0:
                    return

                # Add independent variables, the prior changes
                context.sma15 = get_sma(close=context.recent_prices[security], days=15, window=context.window_length)
                context.sma50 = get_sma(close=context.recent_prices[security], days=50, window=context.window_length)

                context.X = np.array(list(zip(context.sma15, context.sma50)))
                context.Y = context.recent_prices[security]  # Add dependent variable, the final change

                if len(context.Y) >= context.data_points:
                    context.mdl.fit(context.X, context.Y[context.window_length - 1:])  # Generate the model

                    context.pred = context.mdl.predict(context.X[-1:])  # Predict

                    context.forecast = np.append(context.recent_prices[security], context.pred)

                    for k in range(1, context.forecast_steps):
                        context.sma15 = get_sma(close=context.forecast, days=15, window=context.window_length)
                        context.sma50 = get_sma(close=context.forecast, days=50, window=context.window_length)

                        context.X = np.array(list(zip(context.sma15, context.sma50)))

                        context.pred = np.append(context.pred, context.mdl.predict(context.X[-1:]))

                        context.forecast = np.append(context.forecast, context.pred[-1:])

                    # If prediction = 1, buy all shares affordable, if 0 sell all shares
                    if (context.forecast[-1:] - context.pred[0]) > context.forecast_difference:
                        # order(asset=symbol(security), amount=100)
                        order_target_percent(asset=symbol(security),
                                             target=get_percentage_difference(first=context.pred[0],
                                                                              last=context.forecast[-1:]))
                    elif (context.forecast[-1:] - context.pred[0]) < -context.forecast_difference:
                        order_target_percent(asset=symbol(security),
                                             target=-get_percentage_difference(first=context.pred[0],
                                                                               last=context.forecast[-1:]))

    def record_vars(context, data):
        """
        Plot variables at the end of each day.
        """
        record(prediction=int(context.pred[0]))


def get_sma(close, days, window):
    sma = ta.SMA(np.array(close), days)[window - 1:]

    # drop nan values
    sma = sma[~np.isnan(sma)]

    return sma


def get_percentage_difference(first, last):
    percent = ((last - first) / first) * 10

    percent = float(np.around(percent, 2))

    return percent


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
