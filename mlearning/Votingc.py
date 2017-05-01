import functions as fc
import random as rand
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC

df = fc.get_time_series('AAPL')

# add the outcome variable, 1 if the trading session was positive (close>open), 0 otherwise
df['outcome'] = df.apply(lambda x: 1 if x['adj_close'] > x['adj_open'] else -1, axis=1)

df = fc.get_sma_classifier_features(df)

# cross-validation testing
split = rand.uniform(0.60, 0.80)

train_size = int(len(df) * split)

train, test = df[0:train_size], df[train_size:len(df)]

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

confidence = mdl.score(test[features].values, test['outcome'].values)

pred = mdl.predict(test[features].values)

# out-of-sample test
n_steps = 21

forecast = fc.forecast_classifier(model=mdl, sample=test, features=features, steps=n_steps)['outcome']
