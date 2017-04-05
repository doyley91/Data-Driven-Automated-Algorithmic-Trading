import functions as fc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

AAPL = fc.get_time_series('AAPL').asfreq('D', method='ffill').round(2)

fc.end_of_day_plot(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# log returns
lrets = np.log(AAPL['adj_close'] / AAPL['adj_close'].shift(1)).dropna()

fc.plot_time_series(lrets, lags=30)

training_set = AAPL[:-500]
test_set = AAPL[-500:]

lrets_training_set = np.log(training_set['adj_close'] / training_set['adj_close'].shift(1)).dropna()
lrets_test_set = np.log(test_set['adj_close'] / test_set['adj_close'].shift(1)).dropna()

res_tup = fc.get_best_arima_model(lrets_training_set)

res_tup[2].summary()
# aic: -57429.17161 | order: (4, 0, 2)

fc.plot_time_series(res_tup[2].resid, lags=30)

# Create a 21 day forecast of AAPL returns with 95%, 99% CI
n_steps = 21

# out-of-sample prediction
f, err95, ci95 = res_tup[2].forecast(steps=n_steps)  # 95% CI
_, err99, ci99 = res_tup[2].forecast(steps=n_steps, alpha=0.01)  # 99% CI

idx = pd.date_range(AAPL.index[-1], periods=n_steps, freq='D')
fc_95 = pd.DataFrame(np.column_stack([f, ci95]), index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
fc_99 = pd.DataFrame(np.column_stack([ci99]), index=idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_all = fc_95.combine_first(fc_99)

# in-sample prediction
pred = res_tup[2].predict(start=lrets_test_set.index[0]-1, end=lrets_test_set.index[-1]).dropna()

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lrets_test_set, color='blue')
ax.plot(pred)
ax.plot(fc_all)
ax.fill_between(fc_all.index, fc_all.lower_ci_95, fc_all.upper_ci_95, color='gray', alpha=0.7)
ax.fill_between(fc_all.index, fc_all.lower_ci_99, fc_all.upper_ci_99, color='gray', alpha=0.2)
ax.set(title='{} Day Return Forecast\n ARIMA{}'.format(n_steps, res_tup[1]), xlabel='time', ylabel='%')
ax.legend(['Original', 'In-sample prediction', 'Forecast', 'Lower_ci_95', 'Lower_ci_99', 'Upper_ci_95', 'Upper_ci_99'])
fig.tight_layout()