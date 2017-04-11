import functions as fc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

AAPL = fc.get_time_series('AAPL')

fc.plot_end_of_day(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# log returns
AAPL['log_returns'] = np.log(AAPL['adj_close'] / AAPL['adj_close'].shift(1))

AAPL['log_returns'].dropna(inplace=True)

# plotting the histogram of returns
fc.plot_histogram(AAPL['log_returns'])

fc.plot_time_series(AAPL['log_returns'], lags=30)

print("AAPL Series\n"
      "-------------\n"
      "mean: {:.3f}\n"
      "variance: {:.3f}\n"
      "standard deviation: {:.3f}".format(AAPL['adj_close'].mean(), AAPL['adj_close'].var(), AAPL['adj_close'].std()))

adfstat, pvalue, critvalues, resstore, dagostino_results, shapiro_results, ks_results, anderson_results, kpss_results = fc.get_stationarity_statistics(AAPL['log_returns'])

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

# Fit ARMA model to AAPL returns
res_tup = fc.get_best_arma_model(AAPL['log_returns'])

res_tup[2].summary()

fc.plot_time_series(res_tup[2].resid, lags=30)

train_size = int(len(AAPL) * 0.80)

train, test = AAPL[0:train_size], AAPL[train_size:len(AAPL)]

# in-sample prediction
pred = res_tup[2].predict(start=len(train), end=len(train)+len(test)-1)

pred = pred.multiply(10)

results = pd.DataFrame(data=dict(original=test['log_returns'].values, prediction=pred.values), index=test.index)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(results['original'])
ax.plot(results['prediction'])
ax.set(title='ARMA{} In-Sample Return Prediction'.format(res_tup[1]), xlabel='time', ylabel='$')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()

#out-of-sample forecast
n_days = 21

forecast = res_tup[2].forecast(steps=n_days)

forecast = forecast.multiply(10)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(forecast[0])
ax.set(title='{} Day ARMA{} Out-Of-Sample Return Forecast'.format(n_days, res_tup[1]), xlabel='time', ylabel='$')
ax.legend(['Prediction'])
fig.tight_layout()

# Create a 21 day forecast of AAPL returns with 95%, 99% CI
n_steps = 21

f, err95, ci95 = res_tup[2].forecast(steps=n_steps)  # 95% CI
_, err99, ci99 = res_tup[2].forecast(steps=n_steps, alpha=0.01)  # 99% CI

idx = pd.date_range(AAPL.index[-1], periods=n_steps, freq='D')
fc_95 = pd.DataFrame(np.column_stack([f, ci95]), index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
fc_99 = pd.DataFrame(np.column_stack([ci99]), index=idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_all = fc_95.combine_first(fc_99)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = plt.gca()
test.plot(ax=ax, label='AAPL Returns')
pred.plot(ax=ax, style='r-', label='In-sample prediction')
fc_all.plot(ax=ax, style=['b-', '0.2', '0.75', '0.2', '0.75'])
plt.fill_between(fc_all.index, fc_all.lower_ci_95, fc_all.upper_ci_95, color='gray', alpha=0.7)
plt.fill_between(fc_all.index, fc_all.lower_ci_99, fc_all.upper_ci_99, color='gray', alpha=0.2)
plt.title('{} Day AAPL Return Forecast\nARMA{}'.format(n_steps, res_tup[1]))
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
