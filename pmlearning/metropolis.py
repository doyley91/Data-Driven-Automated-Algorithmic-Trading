import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
from pymc3.math import exp

import functions as fc

plt.style.use('ggplot')

AAPL = fc.get_time_series('AAPL')[-500:]

fc.plot_end_of_day(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

AAPL['log_returns'] = np.log(AAPL['adj_close'] / AAPL['adj_close'].shift(1))

AAPL['log_returns'].dropna(inplace=True)

fc.plot_end_of_day(AAPL['log_returns'], title='AAPL', xlabel='time', ylabel='%', legend='Returns %')

train, test = np.arange(0, 450), np.arange(451, len(AAPL['log_returns']))
n = len(train)

with pm.Model() as model:
    sigma = pm.Exponential('sigma', 1. / .02, testval=.1)
    mu = pm.Normal('mu', 0, sd=5, testval=.1)

    nu = pm.Exponential('nu', 1. / 10)
    logs = pm.GaussianRandomWalk('logs', tau=sigma ** -2, shape=n)

    # lam uses variance in pymc3, not sd like in scipy
    r = pm.StudentT('r', nu, mu=mu, lam=1 / exp(-2 * logs), observed=AAPL['log_returns'].values[train])

with model:
    start = pm.find_MAP(vars=[logs], fmin=sp.optimize.fmin_powell)

with model:
    step = pm.Metropolis(vars=[logs, mu, nu, sigma], start=start)
    start2 = pm.sample(100, step, start=start)[-1]

    step = pm.Metropolis(vars=[logs, mu, nu, sigma], start=start2)
    trace = pm.sample(2000, step, start=start2)

pm.traceplot(trace)

sim_returns, vol = fc.generate_proj_returns(1000, trace, len(test))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['log_returns'].values, color='blue')
ax.plot(1 + len(train) + np.arange(0, len(test)), sim_returns[1, :], color='red')
ax.set(title='Returns Forecast', xlabel='time', ylabel='%')
ax.legend(['Original', 'Forecast'])
fig.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(111)
[ax.plot(1 / np.exp(trace[k]['logs']), color='red', alpha=.2) for k in range(1000, len(trace))]
ax.plot(AAPL['log_returns'].values, color='blue')
[ax.plot(1 + len(train) + np.arange(0, len(test)), 1 / np.exp(vol[j, :]), alpha=.01, color='yellow') for j in
 range(0, 1000)]
ax.set_ylim([-.05, .05])
ax.set(title='Volatility Forecast', xlabel='time', ylabel='%')
ax.legend(['Original Returns', 'Original Volatility', 'Forecast Volatility'])
fig.tight_layout()
