'''
Source: https://pymc-devs.github.io/pymc3/notebooks/getting_started.html
'''

import functions as fc
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import scipy as sp

# setting the style of the charts
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
    r = pm.StudentT('r', nu, mu=mu, lam=1 / np.exp(-2 * logs), observed=AAPL['log_returns'].values[train])

with model:
    start = pm.find_MAP(vars=[logs], fmin=sp.optimize.fmin_l_bfgs_b)

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
ax.plot(AAPL['log_returns'].values, color='blue')
ax.plot(1+len(train)+np.arange(0, len(test)), sim_returns[1, :], color='red')
ax.set(title='NUTS In-Sample Returns Prediction', xlabel='time', ylabel='%')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(111)
[ax.plot(1 / np.exp(trace[k]['logs']), color='red', alpha=.2) for k in range(1000, len(trace))]
ax.plot(AAPL['log_returns'].values, color='blue')
[ax.plot(1 + len(train) + np.arange(0, len(test)), 1 / np.exp(vol[j, :]), alpha=.01, color='yellow') for j in
 range(0, 1000)]
ax.set(title='Volatility Forecast', xlabel='time', ylabel='%')
ax.legend(['Original Returns', 'Original Volatility', 'Forecast Volatility'])
fig.tight_layout()

# out-of-sample test
n_steps = 21

sim_returns, vol = fc.generate_proj_returns(1000, trace, len(test) + n_steps)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sim_returns[1, :][-n_steps:])
ax.set(title='NUTS Out-of-Sample Returns Forecast', xlabel='time', ylabel='%')
ax.legend(['Forecast'])
fig.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(111)
[ax.plot(1 / np.exp(trace[k]['logs']), color='red', alpha=.2) for k in range(1000, len(trace))]
ax.plot(AAPL['log_returns'].values, color='blue')
[ax.plot(1 + len(train) + np.arange(0, len(test)), 1 / np.exp(vol[j, :]), alpha=.01, color='yellow') for j in
 range(0, 1000)]
ax.set(title='Volatility Forecast', xlabel='time', ylabel='%')
ax.legend(['Original Returns', 'Original Volatility', 'Forecast Volatility'])
fig.tight_layout()

