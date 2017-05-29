"""
Module Docstring
"""

__author__ = "Gabriel Gauci Maistre"
__version__ = "0.1.0"
__license__ = "MIT"

import random as rand
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import functions as fc


def main(tickers=['AAPL'], start=None, end=None, n_steps=21):
    data = OrderedDict()
    pred_data = OrderedDict()
    forecast_data = OrderedDict()

    for ticker in tickers:
        data[ticker] = fc.get_time_series(ticker, start, end)

        # log_returns
        data[ticker]['log_returns'] = np.log(data[ticker]['adj_close'] / data[ticker]['adj_close'].shift(1))

        data[ticker]['log_returns'].dropna(inplace=True)

        # plotting the histogram of returns
        fc.plot_histogram(y=data[ticker]['log_returns'], ticker=ticker)

        fc.plot_time_series(y=data[ticker]['log_returns'], lags=30, ticker=ticker)

        print("{} Series\n"
              "-------------\n"
              "mean: {:.3f}\n"
              "median: {:.3f}\n"
              "maximum: {:.3f}\n"
              "minimum: {:.3f}\n"
              "variance: {:.3f}\n"
              "standard deviation: {:.3f}\n"
              "skewness: {:.3f}\n"
              "kurtosis: {:.3f}".format(ticker,
                                        data[ticker]['adj_close'].mean(),
                                        data[ticker]['adj_close'].median(),
                                        data[ticker]['adj_close'].max(),
                                        data[ticker]['adj_close'].min(),
                                        data[ticker]['adj_close'].var(),
                                        data[ticker]['adj_close'].std(),
                                        data[ticker]['adj_close'].skew(),
                                        data[ticker]['adj_close'].kurtosis()))

        adfstat, pvalue, critvalues, resstore, dagostino_results, shapiro_results, ks_results, anderson_results, kpss_results = fc.get_stationarity_statistics(
            data[ticker]['log_returns'].values)

        print("{} Stationarity Statistics\n"
              "-------------\n"
              "Augmented Dickey-Fuller unit root test: {}\n"
              "MacKinnon’s approximate p-value: {}\n"
              "Critical values for the test statistic at the 1 %, 5 %, and 10 % levels: {}\n"
              "D’Agostino and Pearson’s normality test: {}\n"
              "Shapiro-Wilk normality test: {}\n"
              "Kolmogorov-Smirnov goodness of fit test: {}\n"
              "Anderson-Darling test: {}\n"
              "Kwiatkowski, Phillips, Schmidt, and Shin (KPSS) stationarity test: {}".format(ticker,
                                                                                             adfstat,
                                                                                             pvalue,
                                                                                             critvalues,
                                                                                             dagostino_results,
                                                                                             shapiro_results,
                                                                                             ks_results,
                                                                                             anderson_results,
                                                                                             kpss_results))

        # Fit MA model to AAPL returns
        res_tup = fc.get_best_ma_model(data[ticker]['log_returns'])

        res_tup[2].summary()

        # verify stationarity
        adfstat, pvalue, critvalues, resstore, dagostino_results, shapiro_results, ks_results, anderson_results, kpss_results = fc.get_stationarity_statistics(
            res_tup[2].resid.values)

        print("Stationarity Statistics\n"
              "-------------\n"
              "Augmented Dickey-Fuller unit root test: {}\n"
              "MacKinnon’s approximate p-value: {}\n"
              "Critical values for the test statistic at the 1 %, 5 %, and 10 % levels: {}\n"
              "D’Agostino and Pearson’s normality test: {}\n"
              "Shapiro-Wilk normality test: {}\n"
              "Kolmogorov-Smirnov goodness of fit test: {}\n"
              "Anderson-Darling test: {}\n"
              "Kwiatkowski, Phillips, Schmidt, and Shin (KPSS) stationarity test: {}".format(adfstat,
                                                                                             pvalue,
                                                                                             critvalues,
                                                                                             dagostino_results,
                                                                                             shapiro_results,
                                                                                             ks_results,
                                                                                             anderson_results,
                                                                                             kpss_results))

        fc.plot_histogram(y=res_tup[2].resid, ticker=ticker, title='MA')

        fc.plot_time_series(y=res_tup[2].resid, lags=30, ticker=ticker, title='MA')

        # cross-validation testing
        split = rand.uniform(0.60, 0.80)

        train_size = int(len(data[ticker]) * split)

        train, test = data[ticker][0:train_size], data[ticker][train_size:len(data[ticker])]

        # in-sample prediction
        pred_data[ticker] = res_tup[2].predict(start=len(train),
                                               end=len(train) + len(test) - 1)

        pred_results = pd.DataFrame(data=dict(original=test['log_returns'],
                                              prediction=pred_data[ticker].values),
                                    index=test.index)

        # prediction plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(pred_results['original'])
        ax.plot(pred_results['prediction'])
        ax.set(title='{} MA{} In-Sample Return Prediction'.format(ticker, res_tup[1]), xlabel='time', ylabel='$')
        ax.legend(['Original', 'Prediction'])
        fig.tight_layout()
        fig.savefig('charts/{}-MA-In-Sample-Return-Prediction.png'.format(ticker))

        # out-of-sample forecast
        forecast_data[ticker] = res_tup[2].forecast(steps=n_steps)

        # forecast plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(forecast_data[ticker][0])
        ax.set(title='{} Day {} MA{} Out-of-Sample Return Forecast'.format(n_steps, ticker, res_tup[1]),
               xlabel='time',
               ylabel='$')
        ax.legend(['Forecast'])
        fig.tight_layout()
        fig.savefig('charts/{}-Day-{}-MA-Out-of-Sample-Return-Forecast.png'.format(n_steps, ticker))

    # end of day plot of all tickers
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ticker in tickers:
        ax.plot(data[ticker]['adj_close'])
    ax.set(title='Time series plot', xlabel='time', ylabel='$')
    ax.legend(tickers)
    fig.tight_layout()
    fig.savefig('charts/stocks.png')

    return forecast_data

if __name__ == '__main__':
    tickers = ['MSFT', 'CDE', 'NAVB', 'HRG', 'HL']

    main(tickers=tickers, start='1990-1-1', end='2017-1-1', n_steps=100)
