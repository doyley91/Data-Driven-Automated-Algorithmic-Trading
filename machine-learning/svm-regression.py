import functions as fc
import pandas as pd
import numpy as np
from sklearn.svm import SVR
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

# fit a Naive Bayes model to the data
mdl = SVR(kernel='linear').fit(X, Y)
print(mdl)

print(mdl.coef_)
print(mdl.intercept_)

# in-sample test
pred = mdl.predict(test[features].values)

metrics.mean_absolute_error(test['adj_close'], pred)
metrics.mean_squared_error(test['adj_close'], pred)
metrics.median_absolute_error(test['adj_close'], pred)
metrics.r2_score(test['adj_close'], pred)

pred_results = pd.DataFrame(data=dict(original=test['adj_close'], prediction=pred), index=test.index)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(pred_results['original'])
ax.plot(pred_results['prediction'])
ax.set(title='Time Series Plot', xlabel='time', ylabel='$')
ax.legend(['Original $', 'Prediction $'])
fig.tight_layout()

# out-of-sample test
n_steps = 21
forecast = fc.forecast_regression(model=mdl, sample=test.copy(), features=features, steps=n_steps)

data = fc.download_data(dataset="WIKI/AAPL",
                        start_date=test.index.shift(n=1, freq='B')[-1],
                        end_date=test.index.shift(n=n_steps-1, freq='B')[-1])

forecast_results = pd.DataFrame(data=dict(original=data['Adj. Close'], forecast=forecast['adj_close']),
                                index=data.index)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(forecast_results['original'])
ax.plot(forecast_results['forecast'])
ax.set(title='{} Day Out-of-Sample Forecast'.format(n_steps), xlabel='time', ylabel='$')
ax.legend(['Original $', 'Forecast $'])
fig.tight_layout()
