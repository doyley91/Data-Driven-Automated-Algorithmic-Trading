import random as rand
from collections import OrderedDict

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import functions as fc


def main(tickers=['AAPL'], start=None, end=None, n_steps=21):
    data = OrderedDict()
    pred_data = OrderedDict()
    forecast_data = OrderedDict()

    for ticker in tickers:
        data[ticker] = fc.get_time_series(ticker, start, end)

        # add the outcome variable, 1 if the trading session was positive (close>open), 0 otherwise
        data[ticker]['outcome'] = data[ticker].apply(lambda x: 1 if x['adj_close'] > x['adj_open'] else -1, axis=1)

        data[ticker] = fc.get_sma_classifier_features(data[ticker])

        # cross-validation testing
        split = rand.uniform(0.60, 0.80)

        train_size = int(len(data[ticker]) * split)

        train, test = data[ticker][0:train_size], data[ticker][train_size:len(data[ticker])]

        features = ['sma_2', 'sma_3', 'sma_4', 'sma_5', 'sma_6']

        # values of features
        X = list(train[features].values)

        # target values
        Y = list(train['outcome'])

        clf1 = AdaBoostClassifier()
        clf2 = RandomForestClassifier()
        clf3 = DecisionTreeClassifier()
        clf4 = KNeighborsClassifier()
        clf5 = LogisticRegression()
        clf6 = SGDClassifier()
        clf7 = MLPClassifier()
        clf8 = GaussianNB()
        clf9 = BernoulliNB()
        clf10 = SVC()

        mdl = VotingClassifier(estimators=[('bt', clf1),
                                           ('rf', clf2),
                                           ('dt', clf3),
                                           ('knn', clf4),
                                           ('lgt', clf5),
                                           ('sgd', clf6),
                                           ('mlp', clf7),
                                           ('gnb', clf8),
                                           ('bnb', clf9),
                                           ('svm', clf10)],
                               voting='hard').fit(X, Y)

        print(mdl)

        confidence = mdl.score(test[features].values, test['outcome'].values)

        print("{} Voting Classifier\n"
              "-------------\n"
              "Confidence: {}\n".format(ticker,
                                        confidence))

        pred = mdl.predict(test[features].values)

        pred_data[ticker] = pred

        # out-of-sample test
        forecast_data[ticker] = fc.forecast_classifier(model=mdl, sample=test, features=features, steps=n_steps)[
            'outcome']

    return forecast_data


if __name__ == '__main__':
    tickers = ['MSFT', 'CDE', 'NAVB', 'HRG', 'HL']

    main(tickers=tickers, start='1990-1-1', end='2017-1-1', n_steps=100)
