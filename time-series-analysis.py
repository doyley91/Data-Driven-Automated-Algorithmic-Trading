'''
Source: http://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016
'''

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# location of the data set
file_location = "data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"

# importing the data set, converting date column to datetime, making the trading date the index for the Pandas DataFrame and sorting the DataFrame by date
df = pd.read_csv(file_location, index_col='date', parse_dates=True)

# creating a DataFrame with just Apple EOD data
AAPL = df.loc[df['ticker'] == "AAPL"]

# retrieving rows where adj_close is finite
AAPL = AAPL[np.isfinite(AAPL['adj_close'])]

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['adj_close'])
ax.set(title='AAPL', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

# log returns
log_returns = np.log(AAPL['adj_close'] / AAPL['adj_close'].shift(1)).dropna()

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(log_returns)
ax.set(title='AAPL Log Returns', xlabel='time', ylabel='%')
ax.legend(['Log Returns'])
fig.tight_layout()

def tsplot(y, lags=None):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    fig = plt.figure()
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    pp_ax = plt.subplot2grid(layout, (2, 1))

    y.plot(ax=ts_ax)
    ts_ax.set_title('Time Series Analysis Plots')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
    sm.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('QQ Plot')
    scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

    plt.tight_layout()

# First difference of simulated Random Walk series
tsplot(np.diff(AAPL['adj_close']), lags=30)

print("AAPL Series\n-------------\nmean: {:.3f}\nvariance: {:.3f}\nstandard deviation: {:.3f}".format(AAPL['adj_close'].mean(),
                                                                                                      AAPL['adj_close'].var(),
                                                                                                      AAPL['adj_close'].std()))

# Select best lag order for AAPL returns
mdl = smt.AR(log_returns).fit(maxlag=30, ic='aic', trend='nc')
mdl.summary()
est_order = smt.AR(log_returns).select_order(maxlag=30, ic='aic', trend='nc')

print('\nalpha estimate: {:3.5f} | best lag order = {}'.format(mdl.params[0], est_order))
# print('best estimated lag order = {}'.format(est_order))
# best estimated lag order = 14

tsplot(mdl.resid, lags=14)

# Create a 21 day forecast of AAPL returns with 95%, 99% CI
n_steps = 21

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = plt.gca()
ts = log_returns[-500:]
ts.plot(ax=ax, label='AAPL Returns')
# in sample prediction
pred = mdl.predict(start=ts.index[0], end=ts.index[-1]).dropna()
pred.plot(ax=ax, style='r-', label='In-sample prediction')
plt.title('{} Day AAPL Return Forecast\nAR{}'.format(n_steps, est_order))
plt.legend(loc='best', fontsize=10)
plt.tight_layout()

# Fit MA(3) to AAPL returns
mdl = smt.ARMA(log_returns, order=(0, 3)).fit(maxlag=30, method='mle', trend='nc')
mdl.summary()
tsplot(mdl.resid, lags=30)

# Fit ARMA model to AAPL returns
best_aic = np.inf
best_order = None
best_mdl = None

rng = range(5) # [0,1,2,3,4,5]
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(log_returns, order=(i, j)).fit(method='mle', trend='nc')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue

print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
# aic: -38112.82032 | order: (4, 4)

best_mdl.summary()
tsplot(best_mdl.resid, lags=30)

# Create a 21 day forecast of AAPL returns with 95%, 99% CI
n_steps = 21

f, err95, ci95 = best_mdl.forecast(steps=n_steps) # 95% CI
_, err99, ci99 = best_mdl.forecast(steps=n_steps, alpha=0.01) # 99% CI

idx = pd.date_range(AAPL.index[-1], periods=n_steps, freq='D')
fc_95 = pd.DataFrame(np.column_stack([f, ci95]), index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
fc_99 = pd.DataFrame(np.column_stack([ci99]), index=idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_all = fc_95.combine_first(fc_99)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = plt.gca()
ts = log_returns[-500:]
ts.plot(ax=ax, label='AAPL Returns')
# in sample prediction
pred = best_mdl.predict(start=ts.index[0], end=ts.index[-1]).dropna()
pred.plot(ax=ax, style='r-', label='In-sample prediction')
plt.title('{} Day AAPL Return Forecast\nARIMA{}'.format(n_steps, best_order))
plt.legend(loc='best', fontsize=10)
plt.tight_layout()

# Fit ARIMA(p, d, q) model to AAPL Returns
# pick best order and final model based on aic

best_aic = np.inf
best_order = None
best_mdl = None

pq_rng = range(5) # [0,1,2,3,4]
d_rng = range(2) # [0,1]
for i in pq_rng:
    for d in d_rng:
        for j in pq_rng:
            try:
                tmp_mdl = smt.ARIMA(log_returns, order=(i, d, j)).fit(method='mle', trend='nc')
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (i, d, j)
                    best_mdl = tmp_mdl
            except: continue

print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
# aic: -38112.82032 | order: (4, 0, 4)

# ARIMA model resid plot
tsplot(best_mdl.resid, lags=30)

# Create a 21 day forecast of AAPL returns with 95%, 99% CI
n_steps = 21

f, err95, ci95 = best_mdl.forecast(steps=n_steps) # 95% CI
_, err99, ci99 = best_mdl.forecast(steps=n_steps, alpha=0.01) # 99% CI

idx = pd.date_range(AAPL.index[-1], periods=n_steps, freq='D')
fc_95 = pd.DataFrame(np.column_stack([f, ci95]), index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
fc_99 = pd.DataFrame(np.column_stack([ci99]), index=idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_all = fc_95.combine_first(fc_99)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = plt.gca()
ts = log_returns[-500:]
ts.plot(ax=ax, label='AAPL Returns')
# in sample prediction
#pred = best_mdl.predict(start=len(log_returns)-500, end=len(log_returns)).dropna()
pred = best_mdl.predict(start=ts.index[0], end=ts.index[-1]).dropna()
pred.plot(ax=ax, style='r-', label='In-sample prediction')
fc_all.plot(ax=ax, style=['b-', '0.2', '0.75', '0.2', '0.75'])
plt.fill_between(fc_all.index, fc_all.lower_ci_95, fc_all.upper_ci_95, color='gray', alpha=0.7)
plt.fill_between(fc_all.index, fc_all.lower_ci_99, fc_all.upper_ci_99, color='gray', alpha=0.2)
plt.title('{} Day AAPL Return Forecast\nARIMA{}'.format(n_steps, best_order))
plt.legend(loc='best', fontsize=10)
plt.tight_layout()

#calculate variance
var = pd.Series.rolling(AAPL['adj_close'], window=30, min_periods=None, freq=None, center=True).var().dropna()

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(var)
ax.set(title='AAPL Variance', xlabel='time', ylabel='%')
ax.legend(['Variance'])
fig.tight_layout()

#ARCH model

# Select best lag order for AAPL returns
mdl = smt.AR(var).fit(maxlag=30, ic='aic', trend='nc')
est_order = smt.AR(var).select_order(maxlag=30, ic='aic', trend='nc')

print('\nalpha estimate: {:3.5f} | best lag order = {}'.format(mdl.params[0], est_order))
# print('best estimated lag order = {}'.format(est_order))
# best estimated lag order = 30

tsplot(mdl.resid, lags=30)

#GARCH model
def _get_best_model(TS):
    best_aic = np.inf
    best_order = None
    best_mdl = None

    pq_rng = range(5) # [0,1,2,3,4]
    d_rng = range(2) # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(TS, order=(i, d, j)).fit(method='mle', trend='nc')
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl

# Notice I've selected a specific time period to run this analysis
res_tup = _get_best_model(log_returns)
# aic: -38112.82032 | order: (4, 0, 4)

tsplot(res_tup[2].resid, lags=30)

#multiplying by 10 due to convergence warnings since we are dealing with very small numbers
log_returns_f = log_returns.multiply(10)

# Now we can fit the arch model using the best fit arima model parameters
# Using student T distribution usually provides better fit
am = arch_model(log_returns_f, p=4, o=0, q=4, dist='StudentsT')
res = am.fit(update_freq=5, disp='off')
res.summary()

tsplot(res.resid, lags=30)

# Create a 21 day forecast of AAPL returns with 95%, 99% CI
n_steps = 21

f, err95, ci95 = res_tup[2].forecast(steps=n_steps) # 95% CI
_, err99, ci99 = res_tup[2].forecast(steps=n_steps, alpha=0.01) # 99% CI

idx = pd.date_range(AAPL.index[-1], periods=n_steps, freq='D')
fc_95 = pd.DataFrame(np.column_stack([f, ci95]), index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
fc_99 = pd.DataFrame(np.column_stack([ci99]), index=idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_all = fc_95.combine_first(fc_99)
fc_all.head()

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = plt.gca()
ts = log_returns[-500:]
ts.plot(ax=ax, label='AAPL Returns')
# in sample prediction
pred = best_mdl.predict(ts.index[0], ts.index[-1])
pred.plot(ax=ax, style='r-', label='In-sample prediction')
fc_all.plot(ax=ax, style=['b-', '0.2', '0.75', '0.2', '0.75'])
plt.fill_between(fc_all.index, fc_all.lower_ci_95, fc_all.upper_ci_95, color='gray', alpha=0.7)
plt.fill_between(fc_all.index, fc_all.lower_ci_99, fc_all.upper_ci_99, color='gray', alpha=0.2)
plt.title('{} Day AAPL Return Forecast\nGARCH{}'.format(n_steps, best_order))
plt.legend(loc='best', fontsize=10)
plt.tight_layout()