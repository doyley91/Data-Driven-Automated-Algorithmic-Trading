import random as rand
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import functions as fc


def main(tickers=['AAPL'], start=None, end=None, n_steps=21):
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

        features = ['sma_15', 'sma_50']

        # values of features
        X = np.array(train[features].values)

        # target values
        Y = np.array(train['adj_close'])

        mdl = DecisionTreeRegressor().fit(X, Y)
        print(mdl)

        '''
        dot_data = export_graphviz(mdl,
                                   out_file=None,
                                   feature_names=list(train[features]),
                                   class_names='outcome',
                                   filled=True,
                                   rounded=True,
                                   special_characters=True)

        graph = pydot.graph_from_dot_data(dot_data)
        graph.write_png("charts/decision-tree-regression.png")
        '''

        pred = mdl.predict(test[features].values)

        # summarize the fit of the model
        explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score = fc.get_regression_metrics(
            test['adj_close'].values, pred)

        print("{} Decision Trees\n"
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
        forecast_data[ticker] = fc.forecast_regression(model=mdl, sample=test.copy(), features=features, steps=n_steps)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(pred_data[ticker]['original'])
        ax.plot(pred_data[ticker]['prediction'])
        ax.set(title='{} Decision Trees In-Sample Prediction'.format(ticker), xlabel='time', ylabel='$')
        ax.legend(['Original $', 'Prediction $'])
        fig.tight_layout()
        fig.savefig('charts/{}-Decision-Trees-In-Sample-Prediction.png'.format(ticker))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(forecast_data[ticker]['adj_close'][-n_steps:])
        ax.set(title='{} Day {} Decision Trees Out-of-Sample Forecast'.format(n_steps, ticker), xlabel='time',
               ylabel='$')
        ax.legend(['Forecast $'])
        fig.tight_layout()
        fig.savefig('charts/{}-Day-{}-Decision-Trees-Out-of-Sample-Forecast'.format(n_steps, ticker))

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
