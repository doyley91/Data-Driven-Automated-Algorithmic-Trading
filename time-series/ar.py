import functions as fc
import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt

AAPL = fc.get_time_series('AAPL').asfreq('D', method='ffill').round(2)

fc.plot_end_of_day(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# log returns
log_returns = np.log(AAPL['adj_close'] / AAPL['adj_close'].shift(1)).dropna()

# plotting the histogram of returns
fc.plot_histogram(log_returns)

fc.plot_time_series(log_returns, lags=30)

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

# Select best lag order for AAPL returns
mdl = smt.AR(train).fit(maxlag=30, ic='aic', trend='nc')
best_order = smt.AR(train).select_order(maxlag=30, ic='aic', trend='nc')

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

# in-sample prediction
pred = mdl.predict(start=test.index[0], end=test.index[-1]).dropna()

results = pd.DataFrame(data=dict(original=test, prediction=pred), index=test.index)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(results['original'])
ax.plot(results['prediction'])
ax.set(title='{} Day AAPL Return Forecast\nAR{}'.format(len(pred), best_order), xlabel='time', ylabel='$')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()

#out-of-sample forecast
n_days = 21

forecast = mdl.predict(start=test.index[-1], end=test.index[-1] + n_days)
