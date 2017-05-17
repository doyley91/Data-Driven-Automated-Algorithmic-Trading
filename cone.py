import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
from pandas_datareader.data import DataReader
from pymc3.math import exp

import functions as fc

np.random.seed(1000000)  # Who wants to be a millionaire

start, end = dt.datetime(2014, 1, 1), dt.datetime(2015, 12, 31)

sp500 = DataReader('^GSPC', 'yahoo', start, end).loc[:, 'Close']

returns = np.log(sp500[1:] / sp500[0:-1].values)  # Calculate log returns
plt.plot(returns)

train, test = np.arange(0, 450), np.arange(451, len(returns))
n = len(train)

with pm.Model() as model:
    sigma = pm.Exponential('sigma', 1. / .02, testval=.1)
    mu = pm.Normal('mu', 0, sd=5, testval=.1)

    nu = pm.Exponential('nu', 1. / 10)
    logs = pm.GaussianRandomWalk('logs', tau=sigma ** -2, shape=n)

    # lam uses variance in pymc3, not sd like in scipy
    r = pm.StudentT('r', nu, mu=mu, lam=1 / exp(-2 * logs), observed=returns.values[train])

with model:
    start = pm.find_MAP(vars=[logs], fmin=sp.optimize.fmin_powell)

with model:
    step = pm.NUTS(vars=[logs, mu, nu, sigma], scaling=start, gamma=.25)
    start2 = pm.sample(100, step, start=start)[-1]

    # Start next run at the last sampled position.
    step = pm.NUTS(vars=[logs, mu, nu, sigma], scaling=start2, gamma=.55)
    trace = pm.sample(2000, step, start=start2)

pm.traceplot(trace)

sim_returns, vol = fc.generate_proj_returns(1000, trace, len(test))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(returns.values, color='blue')
ax.plot(1 + len(train) + np.arange(0, len(test)), sim_returns[1, :], color='red')
ax.set(title='Returns Forecast', xlabel='time', ylabel='%')
ax.legend(['Original', 'Forecast'])
fig.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(111)
[ax.plot(1 / np.exp(trace[k]['logs']), color='red', alpha=.2) for k in range(1000, len(trace))]
ax.plot(returns.values, color='blue')
[ax.plot(1 + len(train) + np.arange(0, len(test)), 1 / np.exp(vol[j, :]), alpha=.01, color='yellow') for j in
 range(0, 1000)]
ax.set_ylim([-.05, .05])
ax.set(title='Volatility Forecast', xlabel='time', ylabel='%')
ax.legend(['Original Returns', 'Original Volatility', 'Forecast Volatility'])
fig.tight_layout()

prices = fc.get_log_prices(sim_returns, sp500, test)

slope = trace[1000:]['mu'].mean()

trend = np.arange(0, len(test)) * slope

ind = np.arange(len(train) + 1, 1 + len(train) + len(test))
ind2 = np.arange(start, 1 + len(train) + len(test))

fig = plt.figure()
ax = fig.add_subplot(111)
[ax.plot(ind, prices[j, :], alpha=.02, color='red') for j in range(0, 1000)]
ax.plot(ind, trend + np.log(sp500.values)[len(train) + 1],
        alpha=1,
        linewidth=2,
        color='black',
        linestyle='dotted')
ax.plot(np.log(sp500)[ind2])
ax.set_ylim([7.4, 7.8])
ax.set_xlim([0, len(sp500) - 2])
ax.set(title='Time Series Plot', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()
