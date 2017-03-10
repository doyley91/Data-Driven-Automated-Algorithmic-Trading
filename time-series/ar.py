import functions as fc
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt

AAPL = fc.return_ticker('AAPL').asfreq('D', method='ffill')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['adj_close'])
ax.set(title='AAPL', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

# log returns
lrets = np.log(AAPL['adj_close'] / AAPL['adj_close'].shift(1)).dropna()

fc.tsplot(lrets, lags=30)

print("AAPL Series\n-------------\nmean: {:.3f}\nvariance: {:.3f}\nstandard deviation: {:.3f}".format(AAPL['adj_close'].mean(),
                                                                                                      AAPL['adj_close'].var(),
                                                                                                      AAPL['adj_close'].std()))

adfstat, pvalue, critvalues, resstore, dagostino_results, shapiro_results, ks_results, anderson_results = fc.test_stationarity(lrets)

# plotting the histogram of returns
fc.histplot(lrets)

training_set = AAPL[:-500]
test_set = AAPL[-500:]

lrets_training_set = np.log(training_set['adj_close'] / training_set['adj_close'].shift(1)).dropna()
lrets_test_set = np.log(test_set['adj_close'] / test_set['adj_close'].shift(1)).dropna()

# Select best lag order for AAPL returns
mdl = smt.AR(lrets_training_set).fit(maxlag=30, ic='aic', trend='nc')
best_order = smt.AR(lrets_training_set).select_order(maxlag=30, ic='aic', trend='nc')

print('\nalpha estimate: {:3.5f} | best lag order = {}'.format(mdl.params[0], best_order))
# print('best estimated lag order = {}'.format(est_order))
# best estimated lag order = 14

adfstat, pvalue, critvalues, resstore, dagostino_results, shapiro_results, ks_results, anderson_results = fc.test_stationarity(mdl.resid)

fc.histplot(mdl.resid)

fc.tsplot(mdl.resid, lags=30)

# in sample prediction
pred = mdl.predict(start=lrets_test_set.index[0]-1, end=lrets_test_set.index[-1]).dropna()

#results = pd.DataFrame(data=dict(original=lrets_test_set, prediction=pred.values), index=lrets_test_set.index)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lrets_test_set)
ax.plot(pred)
ax.set(title='{} Day AAPL Return Forecast\nAR{}'.format(len(pred), best_order), xlabel='time', ylabel='$')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()