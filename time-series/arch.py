import functions as fc
import pandas as pd
from statsmodels.tsa.ar_model import AR
import matplotlib.pyplot as plt

AAPL = fc.get_time_series('AAPL')

fc.plot_end_of_day(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# calculate variance
AAPL['variance'] = pd.Series.rolling(AAPL['adj_close'], window=30, min_periods=None, freq=None, center=True).var()

AAPL['variance'].dropna(inplace=True)

# plotting the histogram of returns
fc.plot_histogram(AAPL['variance'])

fc.plot_time_series(AAPL['variance'], lags=30)

# ARCH model
# Select best lag order for AAPL returns
mdl = AR(AAPL['variance']).fit(maxlag=30, ic='aic', trend='nc')
best_order = AR(AAPL['variance']).select_order(maxlag=30, ic='aic', trend='nc')

print('\nalpha estimate: {:3.5f} | best lag order = {}'.format(mdl.params[0], best_order))

# plotting the histogram of returns
fc.plot_histogram(mdl.resid)

fc.plot_time_series(mdl.resid, lags=30)

train_size = int(len(AAPL) * 0.80)

train, test = AAPL[0:train_size], AAPL[train_size:len(AAPL)]

# in sample prediction
pred = mdl.predict(start=len(train), end=len(train)+len(test)-1)

results = pd.DataFrame(data=dict(original=test['variance'].values, prediction=pred.values), index=test['variance'])

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(results['original'])
ax.plot(results['prediction'])
ax.set(title='ARCH({}) In-Sample Return Prediction'.format(best_order), xlabel='time', ylabel='$')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()

# out-of-sample forecast
n_days = 21

forecast = mdl.predict(start=(len(train) + len(test) - 1), end=(len(train) + len(test) + n_days + 1))

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(forecast)
ax.set(title='{} Day AR({}) Out-Of-Sample Return Forecast'.format(n_days, best_order), xlabel='time', ylabel='$')
ax.legend(['Prediction'])
fig.tight_layout()
