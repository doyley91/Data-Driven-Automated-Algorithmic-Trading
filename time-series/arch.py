import functions as fc
import pandas as pd
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt

AAPL = fc.get_time_series('AAPL').asfreq('D', method='ffill')

fc.plot_end_of_day(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# calculate variance
var = pd.Series.rolling(AAPL['adj_close'], window=30, min_periods=None, freq=None, center=True).var().dropna()

# plotting the histogram of returns
fc.plot_histogram(var)

fc.plot_time_series(var, lags=30)

train_size = int(len(var) * 0.80)

train, test = var[0:train_size], var[train_size:len(var)]

# ARCH model
# Select best lag order for AAPL returns
mdl = smt.AR(train).fit(maxlag=30, ic='aic', trend='nc')
best_order = smt.AR(train).select_order(maxlag=30, ic='aic', trend='nc')

print('\nalpha estimate: {:3.5f} | best lag order = {}'.format(mdl.params[0], best_order))

# plotting the histogram of returns
fc.plot_histogram(var)

fc.plot_time_series(mdl.resid, lags=30)

# in sample prediction
pred = mdl.predict(start=test.index[0], end=test.index[-1]).dropna()

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(test)
ax.plot(pred)
ax.set(title='{} Day AAPL Return Forecast\nAR{}'.format(len(pred), best_order), xlabel='time', ylabel='$')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()