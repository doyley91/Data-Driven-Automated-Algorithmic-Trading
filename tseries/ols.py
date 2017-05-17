import random as rand
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

import functions as fc


def run(tickers='AAPL', start=None, end=None, n_steps=21):
    data = OrderedDict()
    pred_data = OrderedDict()
    forecast_data = OrderedDict()

    for ticker in tickers:
        data[ticker] = fc.get_time_series(ticker, start, end)

        data[ticker] = fc.get_sma_regression_features(data[ticker]).dropna()

        # cross-validation testing
        split = rand.uniform(0.60, 0.80)

        train_size = int(len(data[ticker]) * split)

        train, test = data[ticker][0:train_size], data[ticker][train_size:len(data[ticker])]

        # values of features
        X = list(train['sma_15'].values)

        # target values
        Y = list(train['adj_close'].values)

        mdl = sm.OLS(Y, X).fit()
        print(mdl.summary())

        print(mdl.params)

        print(mdl.bse)

        # in sample prediction
        pred = mdl.predict(test['sma_15'].values)

        # summarize the fit of the model
        explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score = fc.get_regression_metrics(
            test['adj_close'].values, pred)

        print("{} Ordinary Least Squares\n"
              "-------------\n"
              "Explained variance score: {:.3f}\n"
              "Mean absolute error: {:.3f}\n"
              "Mean squared error: {:.3f}\n"
              "Median absolute error: {:.3f}\n"
              "Coefficient of determination: {:.3f}".format(ticker,
                                                            explained_variance_score,
                                                            mean_absolute_error,
                                                            mean_squared_error,
                                                            median_absolute_error,
                                                            r2_score))

        pred_results = pd.DataFrame(data=dict(original=test['adj_close'], prediction=pred), index=test.index)

        pred_data[ticker] = pred_results

        # out-of-sample test
        forecast_data[ticker] = fc.forecast_regression(model=mdl, sample=test, features='sma_15', steps=n_steps)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ticker in tickers:
        ax.plot(data[ticker]['adj_close'])
    ax.set(title='Time series plot', xlabel='time', ylabel='$')
    ax.legend(tickers)
    fig.tight_layout()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ticker in tickers:
        ax.plot(pred_data[ticker]['original'], color='red')
        ax.plot(pred_data[ticker]['prediction'], color='blue')
    ax.set(title='OLS In-Sample Prediction', xlabel='time', ylabel='$')
    ax.legend(['Original $', 'Prediction $'])
    fig.tight_layout()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ticker in tickers:
        ax.plot(forecast_data[ticker]['adj_close'][-n_steps:])
    ax.set(title='{} Day OLS Out-of-Sample Forecast'.format(n_steps), xlabel='time', ylabel='$')
    ax.legend(tickers)
    fig.tight_layout()

    return forecast_data
