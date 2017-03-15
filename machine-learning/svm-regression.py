import functions as fc
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn import metrics
import matplotlib.pyplot as plt

AAPL = fc.return_ticker('AAPL')

fc.end_of_day_plot(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# add the outcome variable, 1 if the trading session was positive (close>open), 0 otherwise
AAPL['outcome'] = AAPL.apply(lambda x: 1 if x['adj_close'] > x['adj_open'] else -1, axis=1)

# distance between Highest and Opening price
AAPL['ho'] = AAPL['adj_high'] - AAPL['adj_open']

# distance between Lowest and Opening price
AAPL['lo'] = AAPL['adj_low'] - AAPL['adj_open']

# difference between Closing price - Opening price
AAPL['gain'] = AAPL['adj_close'] - AAPL['adj_open']

AAPL = fc.get_sma_features(AAPL).dropna()

training_set = AAPL[:-500]
test_set = AAPL[-500:]

# values of features
X = np.array(training_set[['sma_15', 'sma_50']].values)

# target values
Y = list(training_set['adj_close'])

# fit a Naive Bayes model to the data
mdl = SVR(kernel='linear').fit(X, Y)
print(mdl)

# make predictions
pred = mdl.predict(test_set[['sma_15', 'sma_50']].values)

metrics.mean_absolute_error(test_set['adj_close'], pred)
metrics.mean_squared_error(test_set['adj_close'], pred)
metrics.median_absolute_error(test_set['adj_close'], pred)
metrics.r2_score(test_set['adj_close'], pred)

results = pd.DataFrame(data=dict(original=test_set['adj_close'], prediction=pred), index=test_set.index)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(results['original'])
ax.plot(results['prediction'])
ax.set(title='Time Series Plot', xlabel='time', ylabel='$')
ax.legend(['Original $', 'Forecast $'])
fig.tight_layout()