'''
Source: https://pymc-devs.github.io/pymc3/notebooks/getting_started.html
'''

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
from collections import OrderedDict
import functions as fc
from pymc3.math import exp


def main(tickers=['AAPL'], n_steps=21):
    """
    Main entry point of the app
    """
    data = OrderedDict()
    pred_data = OrderedDict()
    forecast_data = OrderedDict()

    for ticker in tickers:
        data[ticker] = fc.get_time_series(ticker)[-500:]

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

        data[ticker]['log_returns'] = np.log(data[ticker]['adj_close'] / data[ticker]['adj_close'].shift(1))

        data[ticker]['log_returns'].dropna(inplace=True)

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

        train, test = np.arange(0, 450), np.arange(451, len(data[ticker]['log_returns']))
        n = len(train)

        with pm.Model() as model:
            sigma = pm.Exponential('sigma', 1. / .02, testval=.1)
            mu = pm.Normal('mu', 0, sd=5, testval=.1)

            nu = pm.Exponential('nu', 1. / 10)
            logs = pm.GaussianRandomWalk('logs', tau=sigma ** -2, shape=n)

            # lam uses variance in pymc3, not sd like in scipy
            r = pm.StudentT('r', nu, mu=mu, lam=1 / exp(-2 * logs), observed=data[ticker]['log_returns'].values[train])

        with model:
            start = pm.find_MAP(vars=[logs], fmin=sp.optimize.fmin_powell)

        with model:
            step = pm.Metropolis(vars=[logs, mu, nu, sigma], start=start)
            start2 = pm.sample(100, step, start=start)[-1]

            step = pm.Metropolis(vars=[logs, mu, nu, sigma], start=start2)
            trace = pm.sample(2000, step, start=start2)

        pred_data[ticker], vol = fc.generate_proj_returns(1000, trace, len(test))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data[ticker]['log_returns'].values, color='blue')
        ax.plot(1 + len(train) + np.arange(0, len(test)), pred_data[ticker][1, :], color='red')
        ax.set(title='{} Metropolis In-Sample Returns Prediction'.format(ticker), xlabel='time', ylabel='%')
        ax.legend(['Original', 'Prediction'])
        fig.tight_layout()
        fig.savefig('charts/{}-Metropolis-In-Sample-Returns-Prediction.png'.format(ticker))

        # out-of-sample test
        forecast_data[ticker], vol = fc.generate_proj_returns(1000, trace, len(test) + n_steps)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(forecast_data[ticker][1, :][-n_steps:])
        ax.set(title='{} Day {} Metropolis Out-of-Sample Returns Forecast'.format(n_steps, ticker), xlabel='time', ylabel='%')
        ax.legend(['Forecast'])
        fig.tight_layout()
        fig.savefig('charts/{}-Day-{}-Metropolis-Out-of-Sample-Returns-Forecast.png'.format(n_steps, ticker))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ticker in tickers:
        ax.plot(data[ticker]['adj_close'])
    ax.set(title='Time series plot', xlabel='time', ylabel='$')
    ax.legend(tickers)
    fig.tight_layout()
    fig.savefig('charts/stocks-close-price.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ticker in tickers:
        ax.plot(data[ticker]['log_returns'])
    ax.set(title='Time series plot', xlabel='time', ylabel='%')
    ax.legend(tickers)
    fig.tight_layout()
    fig.savefig('charts/stocks-close-returns.png')

    return forecast_data

if __name__ == '__main__':
    """ 
    This is executed when run from the command line 
    """
    tickers = ['MSFT', 'CDE', 'NAVB', 'HRG', 'HL']

    main(tickers=tickers, n_steps=100)
