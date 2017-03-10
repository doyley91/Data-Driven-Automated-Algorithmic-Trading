import functions as fc
import pandas as pd
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt

AAPL = fc.return_ticker('AAPL').asfreq('D', method='ffill')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['adj_close'])
ax.set(title='AAPL', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

#calculate variance
var = pd.Series.rolling(AAPL['adj_close'], window=30, min_periods=None, freq=None, center=True).var().dropna()

fc.tsplot(var, lags=30)

training_set = AAPL[:-500]
test_set = AAPL[-500:]

var_training_set = pd.Series.rolling(training_set['adj_close'], window=30, min_periods=None, freq=None, center=True).var().dropna()
var_test_set = pd.Series.rolling(test_set['adj_close'], window=30, min_periods=None, freq=None, center=True).var().dropna()

#ARCH model
# Select best lag order for AAPL returns
mdl = smt.AR(var_training_set).fit(maxlag=30, ic='aic', trend='nc')
best_order = smt.AR(var_training_set).select_order(maxlag=30, ic='aic', trend='nc')

print('\nalpha estimate: {:3.5f} | best lag order = {}'.format(mdl.params[0], best_order))
# print('best estimated lag order = {}'.format(est_order))
# best estimated lag order = 30

fc.tsplot(mdl.resid, lags=30)

# in sample prediction
pred = mdl.predict(start=var_test_set.index[0], end=var_test_set.index[-1]).dropna()

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(var_test_set)
ax.plot(pred)
ax.set(title='{} Day AAPL Return Forecast\nAR{}'.format(len(pred), best_order), xlabel='time', ylabel='$')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()