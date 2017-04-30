import functions as fc
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
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
      "variance: {:.3f}\n"
      "standard deviation: {:.3f}".format(df['adj_close'].mean(), df['adj_close'].var(), df['adj_close'].std()))

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

mdl = ARIMA(df['log_returns'], order=(1, 0, 0)).fit(method='mle', trend='nc', disp=-1)  # -38079.46236032343
res_tup = fc.get_best_arima_model(df['log_returns'])

res_tup[2].summary()

# plotting the histogram of returns
fc.plot_histogram(res_tup[2].resid)

fc.plot_time_series(res_tup[2].resid, lags=30)

train_size = int(len(df) * 0.80)

train, test = df[0:train_size], df[train_size:len(df)]

# in-sample prediction
pred = res_tup[2].predict(start=len(train), end=len(train) + len(test) - 1)

results = pd.DataFrame(data=dict(original=test['log_returns'], prediction=pred.values), index=test.index)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(results['original'])
ax.plot(results['prediction'])
ax.set(title='ARIMA{} In-Sample Return Prediction'.format(res_tup[1]), xlabel='time', ylabel='$')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()

# out-of-sample forecast
n_days = 21

forecast = res_tup[2].forecast(steps=n_days)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(forecast[0])
ax.set(title='{} Day ARIMA Out-Of-Sample Return Forecast'.format(n_days), xlabel='time', ylabel='$')
ax.legend(['Forecast'])
fig.tight_layout()
