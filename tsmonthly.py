import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.api as smt

import functions as fc

AAPL = fc.get_time_series('AAPL')

fc.plot_end_of_day(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# log returns
log_returns = np.log(AAPL['adj_close'] / AAPL['adj_close'].shift(1)).dropna()

# plotting the histogram of returns
fc.plot_histogram(log_returns)

fc.plot_time_series(log_returns, lags=30)

train_size = int(len(log_returns) * 0.80)

train, test = log_returns[0:train_size], log_returns[train_size:len(log_returns)]

# Select best lag order for AAPL returns
mdl = smt.AR(log_returns.values).fit(maxlag=30, ic='aic', trend='nc')
best_order = smt.AR(log_returns).select_order(maxlag=30, ic='aic', trend='nc')

print('alpha estimate: {:3.5f} | best lag order = {}'.format(mdl.params[0], best_order))

fc.plot_time_series(mdl.resid, lags=30)

# in-sample prediction
pred = mdl.predict(start=test.index[0], end=test.index[-1]).dropna()
pred = mdl.predict(start=len(train) + 1, end=len(train) + len(test))

results = pd.DataFrame(data=dict(original=log_returns, prediction=pred), index=test.index)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(results['original'])
ax.plot(results['prediction'])
ax.set(title='In-Sample Return Forecast\nAR{}'.format(best_order), xlabel='time', ylabel='$')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()

# out-of-sample forecast
n_days = 21

forecast = mdl.predict(start=test.index[-1], end=test.index[-1] + n_days)
forecast = mdl.predict(start=len(train) + len(test), end=len(train) + len(test) + n_days - 1)

# out-of-sample forecast
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(forecast)
ax.set(title='{} Day Out-Of-Sample Return Forecast\nAR{}'.format(n_days, best_order), xlabel='time', ylabel='$')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()
