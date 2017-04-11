import functions as fc
import pandas as pd
import numpy as np
from arch import arch_model
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
      "median: {:.3f}\n"
      "maximum: {:.3f}\n"
      "minimum: {:.3f}\n"
      "variance: {:.3f}\n"
      "standard deviation: {:.3f}\n"
      "skewness: {:.3f}\n"
      "kurtosis: {:.3f}".format(AAPL['adj_close'].mean(),
                                AAPL['adj_close'].median(),
                                AAPL['adj_close'].max(),
                                AAPL['adj_close'].min(),
                                AAPL['adj_close'].var(),
                                AAPL['adj_close'].std(),
                                AAPL['adj_close'].skew(),
                                AAPL['adj_close'].kurtosis()))

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

train_size = int(len(AAPL) * 0.80)

train, test = AAPL[0:train_size], AAPL[train_size:len(AAPL)]

# Select best lag order for AAPL returns
mdl = arch_model(AAPL['log_returns']).fit(last_obs=train.index[-1])
mdl.summary()

adfstat, pvalue, critvalues, resstore, dagostino_results, shapiro_results, ks_results, anderson_results, kpss_results = fc.get_stationarity_statistics(mdl.resid)

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

fc.plot_histogram(mdl.resid)

fc.plot_time_series(mdl.resid, lags=30)

# in-sample prediction
pred = mdl.predict(start=len(train), end=len(train)+len(test)-1)
forecasts = mdl.forecast(horizon=5, start=train.index[-1:])
forecasts.variance[train.index[-1]:].plot()

pred = pred.multiply(10)

results = pd.DataFrame(data=dict(original=test['log_returns'].values, prediction=pred.values), index=test.index)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(results['original'])
ax.plot(results['prediction'])
ax.set(title='AR({}) In-Sample Return Prediction'.format(best_order), xlabel='time', ylabel='%')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()

# out-of-sample forecast
n_days = 21

forecast = mdl.forecast(start=(len(train) + len(test) - 1), end=(len(train) + len(test) + n_days + 1))

forecast = forecast.multiply(10)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(forecast)
ax.set(title='{} Day AR({}) Out-Of-Sample Return Forecast'.format(n_days, best_order), xlabel='time', ylabel='$')
ax.legend(['Prediction'])
fig.tight_layout()
