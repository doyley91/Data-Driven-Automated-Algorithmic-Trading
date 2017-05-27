from collections import OrderedDict

import numpy as np

import functions as fc


def main(tickers=['AAPL'], start=None, end=None):
    data = OrderedDict()

    for ticker in tickers:
        data[ticker] = fc.get_time_series(ticker, start, end)

        # log_returns
        data[ticker]['log_returns'] = np.log(data[ticker]['adj_close'] / data[ticker]['adj_close'].shift(1))

        data[ticker]['log_returns'].dropna(inplace=True)

        # plotting the histogram of returns
        fc.plot_histogram(y=data[ticker]['log_returns'], ticker=ticker)

        fc.plot_time_series(y=data[ticker]['log_returns'], lags=30, ticker=ticker)

if __name__ == '__main__':
    tickers = ['MSFT', 'CDE', 'NAVB', 'HRG', 'HL']

    main(tickers=tickers, start='1990-1-1', end='2017-1-1')
