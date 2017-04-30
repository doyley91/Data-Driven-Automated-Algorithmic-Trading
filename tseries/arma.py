import functions as fc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

AAPL = fc.get_time_series('AAPL')

fc.plot_end_of_day(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# log price
AAPL['log_price'] = np.log(AAPL['adj_close'])

# plotting the histogram of returns
fc.plot_histogram(AAPL['log_price'])

fc.plot_time_series(AAPL['log_price'], lags=30)

print("AAPL Series\n"
      "-------------\n"
      "mean: {:.3f}\n"
      "variance: {:.3f}\n"
      "standard deviation: {:.3f}".format(AAPL['adj_close'].mean(), AAPL['adj_close'].var(), AAPL['adj_close'].std()))

adfstat, pvalue, critvalues, resstore, dagostino_results, shapiro_results, ks_results, anderson_results, kpss_results = fc.get_stationarity_statistics(AAPL['log_price'])

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
res_tup = fc.get_best_arma_model(AAPL['log_price'])

res_tup[2].summary()

fc.plot_time_series(res_tup[2].resid, lags=30)

train_size = int(len(AAPL) * 0.80)

train, test = AAPL[0:train_size], AAPL[train_size:len(AAPL)]

# in-sample prediction
pred = res_tup[2].predict(start=len(train), end=len(train)+len(test)-1)

results = pd.DataFrame(data=dict(original=np.exp(test['log_price']), prediction=np.exp(pred)), index=test.index)

# summarize the fit of the model
explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score = fc.get_regression_metrics(results['original'], results['prediction'])

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

forecast = np.exp(forecast[0])

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(forecast)
ax.set(title='{} Day ARMA{} Out-Of-Sample Return Forecast'.format(n_days, res_tup[1]), xlabel='time', ylabel='$')
ax.legend(['Forecast'])
fig.tight_layout()
