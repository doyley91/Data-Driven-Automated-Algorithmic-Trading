import functions as fc
import numpy as np
import scipy as sp
import pymc3 as pm
from pymc3.math import exp
import matplotlib.pyplot as plt

plt.style.use('ggplot')

AAPL = fc.get_time_series('AAPL')[-500:]

fc.end_of_day_plot(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

returns = np.log(AAPL['adj_close'][1:] / AAPL['adj_close'][0:-1].values)

returns = returns[np.isfinite(returns)]

fc.end_of_day_plot(returns, title='AAPL', xlabel='time', ylabel='%', legend='Returns %')

training_set = returns[:-50]
test_set = returns[-50:]

with pm.Model() as model:
    sigma = pm.Exponential('sigma', 1. / .02, testval=.1)
    mu = pm.Normal('mu', 0, sd=5, testval=.1)

    nu = pm.Exponential('nu', 1. / 10)
    logs = pm.GaussianRandomWalk('logs', tau=sigma ** -2, shape=len(training_set))

    # lam uses variance in pymc3, not sd like in scipy
    r = pm.StudentT('r', nu, mu=mu, lam=1 / exp(-2 * logs), observed=training_set)

with model:
    start = pm.find_MAP(vars=[logs], fmin=sp.optimize.fmin_powell)

with model:
    step = pm.NUTS(vars=[logs, mu, nu, sigma], scaling=start, gamma=.25)
    start2 = pm.sample(100, step, start=start)[-1]

    # Start next run at the last sampled position.
    step = pm.NUTS(vars=[logs, mu, nu, sigma], scaling=start2, gamma=.55)
    trace = pm.sample(2000, step, start=start2)

pm.traceplot(trace)

sim_returns, vol = fc.generate_proj_returns(1000, trace, len(test_set))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(returns.values, color='blue')
ax.plot(1+len(training_set)+np.arange(0, len(test_set)), sim_returns[1, :], color='red')
ax.set(title='Returns Forecast', xlabel='time', ylabel='%')
ax.legend(['Original', 'Forecast'])
fig.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(111)
[ax.plot(1 / np.exp(trace[k]['logs']), color='red', alpha=.2) for k in range(1000, len(trace))]
ax.plot(returns.values, color='blue')
[ax.plot(1 + len(training_set) + np.arange(0, len(test_set)), 1 / np.exp(vol[j, :]), alpha=.01, color='yellow') for j in
 range(0, 1000)]
ax.set_ylim([-.05, .05])
ax.set(title='Volatility Forecast', xlabel='time', ylabel='%')
ax.legend(['Original Returns', 'Original Volatility', 'Forecast Volatility'])
fig.tight_layout()

#Convert simulated returns to log-price
prices = fc.convert_prices_to_log(sim_returns, AAPL['adj_close'], test_set)

slope = trace[1000:]['mu'].mean()

trend = np.arange(0, len(test_set)) * slope

ind = np.arange(len(training_set) + 1, 1 + len(training_set) + len(training_set))
ind2 = np.arange(training_set.index[0], 1 + len(training_set) + len(test_set))

fig = plt.figure()
ax = fig.add_subplot(111)
[ax.plot(ind, prices[j, :], alpha=.02, color='red') for j in range(0, 1000)]
ax.plot(ind, trend + np.log(AAPL['adj_close'])[len(training_set) + 1],
        alpha=1,
        linewidth=2,
        color='black',
        linestyle='dotted')
ax.plot(ind2, np.log(AAPL['adj_close'])[ind2])
ax.set_ylim([7.4, 7.8])
ax.set_xlim([training_set.index[0], AAPL['adj_close'].index[:-2]])
ax.set(title='Time Series Plot', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()
