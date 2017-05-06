import functions as fc
import numpy as np
from collections import OrderedDict


def run(tickers='AAPL', start=None, end=None):
    data = OrderedDict()

    for ticker in tickers:
        data[ticker] = fc.get_time_series(ticker, start, end)

        # log_returns
        data[ticker]['log_returns'] = np.log(data[ticker]['adj_close'] / data[ticker]['adj_close'].shift(1))

        data[ticker]['log_returns'].dropna(inplace=True)

        # plotting the histogram of returns
        fc.plot_histogram(data[ticker]['log_returns'])

        fc.plot_time_series(data[ticker]['log_returns'], lags=30)
