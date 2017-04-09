import functions as fc
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

AAPL = fc.get_time_series('AAPL').round(2)

fc.plot_end_of_day(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# add the outcome variable, 1 if the trading session was positive (close>open), 0 otherwise
AAPL['outcome'] = AAPL.apply(lambda x: 1 if x['adj_close'] > x['adj_open'] else -1, axis=1)

AAPL = fc.get_sma_classifier_features(AAPL)
AAPL = fc.get_sma_regression_features(AAPL).dropna()

train_size = int(len(AAPL) * 0.80)

train, test = AAPL[0:train_size], AAPL[train_size:len(AAPL)]

features = ['sma_2', 'sma_3', 'sma_4', 'sma_5', 'sma_6']
features = ['sma_15', 'sma_50']

# values of features
X = list(train[features].values)

# target values
Y = list(train['outcome'])

# fit a Naive Bayes model to the data
mdl = LogisticRegression().fit(X, Y)
print(mdl)

# make predictions
pred = mdl.predict(test[features].values)

# summarize the fit of the model
metrics.mean_absolute_error(test['outcome'], pred)
metrics.mean_squared_error(test['outcome'], pred)
metrics.median_absolute_error(test['outcome'], pred)
metrics.r2_score(test['outcome'], pred)

results = pd.DataFrame(data=dict(original=test['outcome'], prediction=pred), index=test.index)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(results['original'])
ax.plot(results['prediction'])
ax.set(title='Time Series Plot', xlabel='time', ylabel='$')
ax.legend(['Original $', 'Forecast $'])
fig.tight_layout()