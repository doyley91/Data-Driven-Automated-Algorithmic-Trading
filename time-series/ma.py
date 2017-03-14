import functions as fc
import numpy as np
import matplotlib.pyplot as plt

AAPL = fc.return_ticker('AAPL').asfreq('D', method='ffill')

fc.end_of_day_plot(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# log returns
lrets = np.log(AAPL['adj_close'] / AAPL['adj_close'].shift(1)).dropna()

# verify stationarity
adfstat, pvalue, critvalues, resstore, dagostino_results, shapiro_results, ks_results, anderson_results = fc.test_stationarity(lrets)

# plotting the histogram of returns
fc.histplot(lrets)

fc.tsplot(lrets, lags=30)

training_set = AAPL[:-500]
test_set = AAPL[-500:]

lrets_training_set = np.log(training_set['adj_close'] / training_set['adj_close'].shift(1)).dropna()
lrets_test_set = np.log(test_set['adj_close'] / test_set['adj_close'].shift(1)).dropna()

# Fit MA model to AAPL returns
res_tup = fc.get_best_ma_model(lrets_training_set)
# aic: -38106.64821 | order: (0, 4)

res_tup[2].summary()

# verify stationarity
adfstat, pvalue, critvalues, resstore, dagostino_results, shapiro_results, ks_results, anderson_results = fc.test_stationarity(res_tup[2].resid)

fc.histplot(res_tup[2].resid)

fc.tsplot(res_tup[2].resid, lags=30)

# in sample prediction
pred = res_tup[2].predict(start=lrets_test_set.index[0]-1, end=lrets_test_set.index[-1]).dropna()

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lrets_test_set)
ax.plot(pred)
ax.set(title='{} Day AAPL Return Forecast\nMA{}'.format(len(pred), res_tup[1]), xlabel='time', ylabel='$')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()