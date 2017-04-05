import functions as fc
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

AAPL = fc.get_time_series('AAPL').round(2)

fc.end_of_day_plot(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

AAPL = fc.get_sma_regression_features(AAPL).dropna()

training_set = AAPL[:-500]
test_set = AAPL[-500:]

features = ['sma_15', 'sma_50']

# values of features
X = np.array(training_set[features].values)

# target values
Y = list(training_set['adj_close'])

# fit a Naive Bayes model to the data
mdl = SGDRegressor(loss="epsilon_insensitive").fit(X, Y)
print(mdl)

# make predictions
pred = mdl.predict(test_set[features].values).round(2)

# summarize the fit of the model
metrics.explained_variance_score(test_set['adj_close'], pred)
metrics.mean_absolute_error(test_set['adj_close'], pred)
metrics.mean_squared_error(test_set['adj_close'], pred)
metrics.median_absolute_error(test_set['adj_close'], pred)
metrics.r2_score(test_set['adj_close'], pred)

# in-sample test
results = pd.DataFrame(data=dict(original=test_set['adj_close'], prediction=pred), index=test_set.index)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(results['original'])
ax.plot(results['prediction'])
ax.set(title='Time Series Plot', xlabel='time', ylabel='$')
ax.legend(['Original $', 'Forecast $'])
fig.tight_layout()

# out-of-sample test
n_steps = 21
forecast = fc.forecast_regression(model=mdl, sample=test_set, features=features, steps=n_steps)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(forecast['adj_close'][-n_steps:])
ax.set(title='{} Day Out-of-Sample Forecast'.format(n_steps), xlabel='time', ylabel='$')
ax.legend(['Forecast $'])
fig.tight_layout()
