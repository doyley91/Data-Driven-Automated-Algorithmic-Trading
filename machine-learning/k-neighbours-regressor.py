import functions as fc
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

AAPL = fc.get_time_series('AAPL').round(2)

fc.plot_end_of_day(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

AAPL = fc.get_sma_regression_features(AAPL).dropna()

train_size = int(len(AAPL) * 0.80)

train, test = AAPL[0:train_size], AAPL[train_size:len(AAPL)]

features = ['sma_15', 'sma_50']

# values of features
X = np.array(train[features].values)

# target values
Y = list(train['adj_close'])

# fit a k-nearest neighbor model to the data
mdl = KNeighborsRegressor(n_neighbors=2).fit(X, Y)
print(mdl)

# make predictions
pred = mdl.predict(test[features].values)

metrics.mean_absolute_error(test['adj_close'], pred)
metrics.mean_squared_error(test['adj_close'], pred)
metrics.median_absolute_error(test['adj_close'], pred)
metrics.r2_score(test['adj_close'], pred)

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
