import functions as fc
import random as rand
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def run(ticker='AAPL', start=None, end=None):

    df = fc.get_time_series(ticker, start, end)

    fc.plot_end_of_day(df['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

    df = fc.get_sma_regression_features(df).dropna()

    # cross-validation testing
    split = rand.uniform(0.60, 0.80)

    train_size = int(len(df) * split)

    train, test = df[0:train_size], df[train_size:len(df)]

    features = ['sma_15', 'sma_50']

    # values of features
    X = np.array(train[features].values)

    # target values
    Y = list(train['adj_close'])

    # fit a Naive Bayes model to the data
    mdl = RandomForestRegressor(criterion='mae').fit(X, Y)
    print(mdl)

    # make predictions
    pred = mdl.predict(test[features].values)

    # summarize the fit of the model
    explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score = fc.get_regression_metrics(test['adj_close'].values, pred)

    results = pd.DataFrame(data=dict(original=test['adj_close'], prediction=pred), index=test.index)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(results['original'])
    ax.plot(results['prediction'])
    ax.set(title='Time Series Plot', xlabel='time', ylabel='$')
    ax.legend(['Original $', 'Forecast $'])
    fig.tight_layout()

    # out-of-sample test
    n_steps = 21
    forecast = fc.forecast_regression(model=mdl, sample=test, features=features, steps=n_steps)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(forecast['adj_close'][-n_steps:])
    ax.set(title='{} Day Out-of-Sample Forecast'.format(n_steps), xlabel='time', ylabel='$')
    ax.legend(['Forecast $'])
    fig.tight_layout()

    return forecast['adj_close'][-n_steps:]
