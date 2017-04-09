import functions as fc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

AAPL = fc.get_time_series('AAPL').asfreq('D', method='ffill').round(2)

fc.plot_end_of_day(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# log returns
log_returns = np.log(AAPL['adj_close'] / AAPL['adj_close'].shift(1)).dropna()

# plotting the histogram of returns
fc.plot_histogram(log_returns)

fc.plot_time_series(np.diff(AAPL['adj_close']), lags=30)

print("AAPL Series\n"
      "-------------\n"
      "mean: {:.3f}\n"
      "variance: {:.3f}\n"
      "standard deviation: {:.3f}".format(AAPL['adj_close'].mean(), AAPL['adj_close'].var(), AAPL['adj_close'].std()))

adfstat, pvalue, critvalues, resstore, dagostino_results, shapiro_results, ks_results, anderson_results, kpss_results = fc.get_stationarity_statistics(log_returns)

print("Stationarity Statistics\n"
      "-------------\n"
      "Augmented Dickey-Fuller unit root test: {}\n"
      "MacKinnon’s approximate p-value: {}\n"
      "Critical values for the test statistic at the 1 %, 5 %, and 10 % levels: {}\n"
      "D’Agostino and Pearson’s normality test: {}\n"
      "Shapiro-Wilk normality test: {}\n"
      "Kolmogorov-Smirnov goodness of fit test: {}\n"
      "Anderson-Darling test: {}\n"
      "Kwiatkowski, Phillips, Schmidt, and Shin (KPSS) stationarity test: {}".format(adfstat,
                                                                                     pvalue,
                                                                                     critvalues,
                                                                                     dagostino_results,
                                                                                     shapiro_results,
                                                                                     ks_results,
                                                                                     anderson_results,
                                                                                     kpss_results))

train_size = int(len(log_returns) * 0.80)

train, test = log_returns[0:train_size], log_returns[train_size:len(log_returns)]

#SARIMAX model
res_tup = fc.get_best_sarimax_model(train)

res_tup[2].summary()

# plotting the histogram of returns
fc.plot_histogram(log_returns)

fc.plot_time_series(res_tup[2].resid, lags=30)

# Create a 21 day forecast of AAPL returns with 95%, 99% CI
n_steps = 21

f, err95, ci95 = res_tup[2].forecast(steps=n_steps) # 95% CI
_, err99, ci99 = res_tup[2].forecast(steps=n_steps, alpha=0.01) # 99% CI

idx = pd.date_range(AAPL.index[-1], periods=n_steps, freq='D')
fc_95 = pd.DataFrame(np.column_stack([f, ci95]), index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
fc_99 = pd.DataFrame(np.column_stack([ci99]), index=idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_all = fc_95.combine_first(fc_99)

# in-sample prediction
pred = res_tup[2].predict(start=test.index[0]-1, end=test.index[-1]).dropna()

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = plt.gca()
ts = log_returns[-500:]
ts.plot(ax=ax, label='AAPL Returns')
# in sample prediction
pred = res_tup[2].predict(start=ts.index[0], end=ts.index[-1]).dropna()
pred.plot(ax=ax, style='r-', label='In-sample prediction')
plt.title('{} Day AAPL Return Forecast\nARIMA{}'.format(n_steps, res_tup[1]))
plt.legend(loc='best', fontsize=10)
plt.tight_layout()