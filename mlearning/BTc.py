from collections import OrderedDict

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

import functions as fc


def main(tickers=['AAPL'], start=None, end=None, n_steps=21):
    data = OrderedDict()
    pred_data = OrderedDict()
    forecast_data = OrderedDict()

    for ticker in tickers:
        data[ticker] = fc.get_time_series(ticker, start, end)

        # add the outcome variable, 1 if the trading session was positive (close>open), 0 otherwise
        data[ticker]['outcome'] = data[ticker].apply(lambda x: 1 if x['adj_close'] > x['adj_open'] else 0, axis=1)

        data[ticker] = fc.get_sma_classifier_features(data[ticker])

        train_size = int(len(data[ticker]) * 0.80)

        train, test = data[ticker][0:train_size], data[ticker][train_size:len(data[ticker])]

        features = ['sma_2', 'sma_3', 'sma_4', 'sma_5', 'sma_6']

        # values of features
        X = list(train[features].values)

        # target values
        Y = list(train['outcome'])

        # fit a Naive Bayes model to the data
        mdl = AdaBoostClassifier().fit(X, Y)
        print(mdl)

        # in-sample prediction
        pred = mdl.predict(test[features].values)

        # summarize the fit of the model
        classification_report, confusion_matrix = fc.get_classifier_metrics(test['outcome'].values, pred)

        print("{} Boosted Tree\n"
              "-------------\n"
              "Classification report: {}\n\n"
              "Confusion matrix: {}\n\n".format(ticker,
                                                classification_report,
                                                confusion_matrix))

        pred_results = pd.DataFrame(data=dict(original=test['outcome'], prediction=pred), index=test.index)

        pred_data[ticker] = pred_results

        # out-of-sample test
        forecast_data[ticker] = fc.forecast_classifier(model=mdl, sample=test, features=features, steps=n_steps)

    return forecast_data


if __name__ == '__main__':
    tickers = ['MSFT', 'CDE', 'NAVB', 'HRG', 'HL']

    main(tickers=tickers, start='1990-1-1', end='2017-1-1', n_steps=100)
