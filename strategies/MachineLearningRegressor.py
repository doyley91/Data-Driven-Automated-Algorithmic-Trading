import logbook as log
import sys
from collections import OrderedDict, deque

import numpy as np
import pandas as pd
import pyfolio as pf
import talib as ta
from sklearn.ensemble import RandomForestRegressor
from zipline.algorithm import TradingAlgorithm
from zipline.api import symbol, order_target_percent, record

import functions as fc


class MachineLearningRegressor(TradingAlgorithm):
    def initialize(context):
        """
        Called once at the start of the algorithm.
        The initialize function is the place to set your tradable universe and define any parameters
        """
        # set_slippage(slippage.FixedSlippage(spread=0.00))
        # set_commission(commission.PerShare(cost=0, min_trade_cost=0))

        context.securities = tickers

        # set_benchmark(symbol(context.securities['SP500']))

        context.window_length = 50  # Amount of prior bars to study

        context.data_points = 100  # There needs to be enough data points to make a good model

        context.pred_steps = 100  # Number of days to forecast

        context.trading_freq = 50  # trading frequency, days

        context.forecast_difference = 10  # forecast increase to invest in

        context.mdl = RandomForestRegressor()  # Use a random forest regressor

        context.recent_prices = OrderedDict()  # Stores recent prices

        context.invested = OrderedDict()

        for security in context.securities:
            context.recent_prices[security] = []
            context.invested[security] = False

        context.sma15 = context.sma50 = []  # Stores the 15 and 50 day simple moving average

        context.X = []  # Independent, or input variables
        context.Y = []  # Dependent, or output variable

        context.pred = deque(maxlen=context.pred_steps - 1)  # Stores most recent prediction

        # schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())

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
                # if len(context.recent_prices[security]) % context.trading_freq != 0.0:
                #   return

                # Add independent variables, the prior changes
                context.sma15 = get_sma(close=context.recent_prices[security], days=15, window=context.window_length)
                context.sma50 = get_sma(close=context.recent_prices[security], days=50, window=context.window_length)

                context.X = np.array(list(zip(context.sma15, context.sma50)))
                context.Y = context.recent_prices[security]  # Add dependent variable, the final change

                if len(context.Y) >= context.data_points:
                    context.mdl.fit(context.X, context.Y[context.window_length - 1:])  # Generate the model

                    for k in range(1, context.pred_steps):
                        context.pred.append(context.mdl.predict(context.X[-1:]))  # Predict

                        context.sma15 = get_sma(close=np.append(context.recent_prices[security],
                                                                context.pred),
                                                days=15,
                                                window=context.window_length)

                        context.sma50 = get_sma(close=np.append(context.recent_prices[security],
                                                                context.pred),
                                                days=50,
                                                window=context.window_length)

                        context.X = np.array(list(zip(context.sma15, context.sma50)))

                    # If prediction goes up by a certain amount buy, else short
                    if (context.pred[-1] - context.pred[0]) > context.forecast_difference:
                        if not context.invested[security]:
                            order_target_percent(asset=symbol(security),
                                                 target=get_percentage_difference(first=context.pred[0],
                                                                                  last=context.pred[-1]))
                            context.invested[security] = True
                    elif (context.pred[-1] - context.pred[0]) < -context.forecast_difference:
                        if context.invested[security]:
                            order_target_percent(asset=symbol(security),
                                                 target=-get_percentage_difference(first=context.pred[0],
                                                                                   last=context.pred[-1]))
                            context.invested[security] = False

    def record_vars(context, data):
        """
        Plot variables at the end of each day.
        """
        record(prediction=int(context.pred[0]))


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
    zipline_logging = log.NestedSetup([
        log.NullHandler(level=log.DEBUG),
        log.StreamHandler(sys.stdout, level=log.INFO),
        log.StreamHandler(sys.stderr, level=log.ERROR),
    ])
    zipline_logging.push_application()

    # tickers to pass to the algorithm
    tickers = ['AAPL', 'MSFT']

    # initialising an ordered dictionary to store all our stocks
    data = OrderedDict()

    # tidying the data for the backtester
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

    # converting data frame data into panel
    panel = pd.Panel(data)

    # initialise strategy class
    algo = MachineLearningRegressor()

    # run strategy
    results = algo.run(panel)

    # get the returns, positions, and transactions from the zipline backtest object
    returns, positions, transactions, gross_lev = pf.utils.extract_rets_pos_txn_from_zipline(results)

    # plot of the top 5 drawdown periods
    pf.plot_drawdown_periods(returns, top=5).set_xlabel('Date')

    # create a full tear sheet for our algorithm. As an example, set the live start date to something arbitrary
    pf.create_full_tear_sheet(returns, positions=positions, transactions=transactions,
                              gross_lev=gross_lev, live_start_date='2009-10-22', round_trips=True)
