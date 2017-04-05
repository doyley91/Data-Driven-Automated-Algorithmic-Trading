import functions as fc
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

AAPL = fc.get_time_series('AAPL')

fc.end_of_day_plot(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# log returns
lrets = np.log(AAPL['adj_close'] / AAPL['adj_close'].shift(1)).dropna()

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lrets)
ax.set(title='AAPL Log Returns', xlabel='time', ylabel='%')
ax.legend(['Log Returns'])
fig.tight_layout()

fc.tsplot(np.diff(AAPL['adj_close']), lags=30)

print("AAPL Series\n-------------\nmean: {:.3f}\nvariance: {:.3f}\nstandard deviation: {:.3f}".format(AAPL['adj_close'].mean(),
                                                                                                      AAPL['adj_close'].var(),
                                                                                                      AAPL['adj_close'].std()))

#SARIMAX model
res_tup = fc.get_best_sarimax_model(lrets)
# aic: -38112.82032 | order: (4, 0, 0)

fc.tsplot(res_tup[2].resid, lags=30)

# Create a 21 day forecast of AAPL returns with 95%, 99% CI
n_steps = 21

f, err95, ci95 = res_tup[2].forecast(steps=n_steps) # 95% CI
_, err99, ci99 = res_tup[2].forecast(steps=n_steps, alpha=0.01) # 99% CI

idx = pd.date_range(AAPL.index[-1], periods=n_steps, freq='D')
fc_95 = pd.DataFrame(np.column_stack([f, ci95]), index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
fc_99 = pd.DataFrame(np.column_stack([ci99]), index=idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_all = fc_95.combine_first(fc_99)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = plt.gca()
ts = lrets[-500:]
ts.plot(ax=ax, label='AAPL Returns')
# in sample prediction
pred = res_tup[2].predict(start=ts.index[0], end=ts.index[-1]).dropna()
pred.plot(ax=ax, style='r-', label='In-sample prediction')
plt.title('{} Day AAPL Return Forecast\nARIMA{}'.format(n_steps, res_tup[1]))
plt.legend(loc='best', fontsize=10)
plt.tight_layout()