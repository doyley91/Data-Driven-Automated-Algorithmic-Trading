import functions as fc
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

AAPL = fc.get_time_series('AAPL')

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

#GARCH model
res_tup = fc.get_best_garch_model(log_returns)

# plotting the histogram of returns
fc.plot_histogram(res_tup[2].resid)

fc.plot_time_series(res_tup[2].resid, lags=30)

#multiplying by 10 due to convergence warnings since we are dealing with very small numbers
log_returns_f = log_returns.multiply(10)

# Now we can fit the arch model using the best fit arima model parameters
# Using student T distribution usually provides better fit
am = arch_model(log_returns_f, p=4, o=0, q=4, dist='StudentsT')
res = am.fit(update_freq=5, disp='off')
res.summary()

# plotting the histogram of returns
fc.plot_histogram(log_returns)

fc.plot_time_series(res.resid, lags=30)

train_size = int(len(log_returns) * 0.80)

train, test = log_returns[0:train_size], log_returns[train_size:len(log_returns)]

# in sample prediction
pred = res_tup[2].predict(start=len(train), end=len(train)+len(test)-1)

results = pd.DataFrame(data=dict(original=test, prediction=pred), index=test.index)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(results['original'])
ax.plot(results['prediction'])
ax.set(title='In-Sample Return Prediction\nGARCH({})'.format(res_tup[1]), xlabel='time', ylabel='$')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()
