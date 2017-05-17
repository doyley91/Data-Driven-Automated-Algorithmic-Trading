import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

import functions as fc

AAPL = fc.get_time_series('AAPL')

fc.plot_end_of_day(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# log price
AAPL['log_price'] = np.log(AAPL['adj_close'])

# plotting the histogram of returns
fc.plot_histogram(AAPL['log_price'])

fc.plot_time_series(AAPL['log_price'], lags=30)

y = AAPL['log_price']

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

for param in pdq:
    try:
        mod = ARIMA(y, order=param)

        results = mod.fit(disp=-1)

        print('ARIMA{} - AIC:{}'.format(param, results.aic))
    except:
        continue

mdl = ARIMA(y, order=(1, 0, 0)).fit(disp=-1)

mdl.summary()

# plotting the histogram of returns
fc.plot_histogram(mdl.resid)

fc.plot_time_series(mdl.resid, lags=30)

train_size = int(len(AAPL) * 0.80)

train, test = AAPL[0:train_size], AAPL[train_size:len(AAPL)]

# in-sample prediction
pred = mdl.predict(start=len(train), end=len(train) + len(test) - 1)

results = pd.DataFrame(data=dict(original=np.exp(test['log_price']), prediction=np.exp(pred)), index=test.index)

# summarize the fit of the model
explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score = fc.get_regression_metrics(
    results['original'], results['prediction'])

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(results['original'])
ax.plot(results['prediction'])
ax.set(title='ARIMA{} In-Sample Return Prediction'.format(res_tup[1]), xlabel='time', ylabel='$')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()

# out-of-sample forecast
n_days = 21

forecast = res_tup[2].forecast(steps=n_days)

forecast = np.exp(forecast[0])

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(forecast)
ax.set(title='{} Day ARIMA Out-Of-Sample Return Forecast'.format(n_days), xlabel='time', ylabel='$')
ax.legend(['Forecast'])
fig.tight_layout()
