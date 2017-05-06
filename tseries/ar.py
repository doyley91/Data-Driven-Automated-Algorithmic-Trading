import functions as fc
import random as rand
import pandas as pd
import numpy as np
from collections import OrderedDict
from statsmodels.tsa.ar_model import AR
import matplotlib.pyplot as plt


def run(tickers='AAPL', start=None, end=None, n_steps=21):
    data = OrderedDict()
    pred_data = OrderedDict()
    forecast_data = OrderedDict()

    for ticker in tickers:
        data[ticker] = fc.get_time_series(ticker, start, end)

        # log_returns
        data[ticker]['log_returns'] = np.log(data[ticker]['adj_close'] / data[ticker]['adj_close'].shift(1))

        data[ticker]['log_returns'].dropna(inplace=True)

        # plotting the histogram of returns
        fc.plot_histogram(data[ticker]['log_returns'])

        fc.plot_time_series(data[ticker]['log_returns'], lags=30)

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

        # Select best lag order for AAPL returns
        mdl = AR(endog=data[ticker]['log_returns']).fit(maxlag=30, ic='aic', trend='nc')
        best_order = AR(data[ticker]['log_returns']).select_order(maxlag=30, ic='aic', trend='nc')

        print('alpha estimate: {:3.5f} | best lag order = {}'.format(mdl.params[0], best_order))

        adfstat, pvalue, critvalues, resstore, dagostino_results, shapiro_results, ks_results, anderson_results, kpss_results = fc.get_stationarity_statistics(
            mdl.resid.values)

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

        fc.plot_histogram(mdl.resid)

        fc.plot_time_series(mdl.resid, lags=30)

        # cross-validation testing
        split = rand.uniform(0.60, 0.80)

        train_size = int(len(data[ticker]) * split)

        train, test = data[ticker][0:train_size], data[ticker][train_size:len(data[ticker])]

        # in-sample prediction
        pred_data[ticker] = mdl.predict(start=len(train),
                                        end=len(train) + len(test) - 1)

        pred_results = pd.DataFrame(data=dict(original=test['log_returns'],
                                              prediction=pred_data[ticker].values),
                                    index=test.index)

        # prediction plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(pred_results['original'], color='red')
        ax.plot(pred_results['prediction'], color='blue')
        ax.set(title='{} AR({}) In-Sample Return Prediction'.format(ticker, best_order), xlabel='time', ylabel='%')
        ax.legend(['Original $', 'Prediction $'])
        fig.tight_layout()

        # out-of-sample forecast
        forecast_data[ticker] = mdl.predict(start=(len(train) + len(test) - 2),
                                            end=(len(train) + len(test) + n_steps + 1))

        # forecast plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(forecast_data[ticker][-n_steps:])
        ax.set(title='{} Day {} AR({}) Out-Of-Sample Return Forecast'.format(n_steps, ticker, best_order),
               xlabel='time',
               ylabel='$')
        ax.legend(tickers)
        fig.tight_layout()

    # end of day plot of all tickers
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ticker in tickers:
        ax.plot(data[ticker]['adj_close'])
    ax.set(title='Time series plot', xlabel='time', ylabel='$')
    ax.legend(tickers)
    fig.tight_layout()

    return forecast_data
