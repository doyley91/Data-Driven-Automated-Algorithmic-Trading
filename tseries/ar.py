import functions as fc
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AR
import matplotlib.pyplot as plt

df = fc.get_time_series('AAPL')

fc.plot_end_of_day(df['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# log_returns
df['log_returns'] = np.log(df['adj_close'] / df['adj_close'].shift(1))

df['log_returns'].dropna(inplace=True)

# plotting the histogram of returns
fc.plot_histogram(df['log_returns'])

fc.plot_time_series(df['log_returns'], lags=30)

print("AAPL Series\n"
      "-------------\n"
      "mean: {:.3f}\n"
      "median: {:.3f}\n"
      "maximum: {:.3f}\n"
      "minimum: {:.3f}\n"
      "variance: {:.3f}\n"
      "standard deviation: {:.3f}\n"
      "skewness: {:.3f}\n"
      "kurtosis: {:.3f}".format(df['adj_close'].mean(),
                                df['adj_close'].median(),
                                df['adj_close'].max(),
                                df['adj_close'].min(),
                                df['adj_close'].var(),
                                df['adj_close'].std(),
                                df['adj_close'].skew(),
                                df['adj_close'].kurtosis()))

adfstat, pvalue, critvalues, resstore, dagostino_results, shapiro_results, ks_results, anderson_results, kpss_results = fc.get_stationarity_statistics(df['log_returns'])

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

# Select best lag order for AAPL returns
mdl = AR(endog=df['log_returns']).fit(maxlag=30, ic='aic', trend='nc')
best_order = AR(df['log_returns']).select_order(maxlag=30, ic='aic', trend='nc')

print('alpha estimate: {:3.5f} | best lag order = {}'.format(mdl.params[0], best_order))

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

train_size = int(len(df) * 0.80)

train, test = df[0:train_size], df[train_size:len(df)]

# in-sample prediction
pred = mdl.predict(start=len(train), end=len(train) + len(test) - 1)

results = pd.DataFrame(data=dict(original=test['log_returns'], prediction=pred.values), index=test.index)

# summarize the fit of the model
explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score = fc.get_regression_metrics(results['original'], results['prediction'])

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

forecast = mdl.predict(start=(len(train) + len(test) - 2), end=(len(train) + len(test) + n_days + 1))

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(forecast)
ax.set(title='{} Day AR({}) Out-Of-Sample Return Forecast'.format(n_days, best_order), xlabel='time', ylabel='$')
ax.legend(['Forecast'])
fig.tight_layout()
