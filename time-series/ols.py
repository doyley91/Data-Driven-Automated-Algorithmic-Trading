import functions as fc
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

AAPL = fc.get_time_series('AAPL').asfreq('D', method='ffill')

fc.end_of_day_plot(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# log returns
lrets = np.log(AAPL['adj_close'] / AAPL['adj_close'].shift(1)).dropna()

fc.tsplot(lrets, lags=30)

training_set = AAPL[:-500]
test_set = AAPL[-500:]

lrets_training_set = np.log(training_set['adj_close'] / training_set['adj_close'].shift(1)).dropna()
lrets_test_set = np.log(test_set['adj_close'] / test_set['adj_close'].shift(1)).dropna()

mdl = sm.OLS(lrets_training_set, sm.add_constant(range(len(lrets_training_set.index)), prepend=True)).fit()
mdl.summary()

fc.tsplot(mdl.resid, lags=30)

# in sample prediction
pred = mdl.predict(sm.add_constant(range(len(lrets_training_set.index))))

results = pd.DataFrame(data=dict(original=AAPL['adj_close'], prediction=pred), index=lrets_training_set.index)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['adj_close'])
ax.plot(pred)
ax.set(title='{} Day AAPL Return Forecast\nAR{}'.format(len(pred)), xlabel='time', ylabel='$')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()