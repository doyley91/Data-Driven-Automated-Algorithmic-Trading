'''
Source: http://srome.github.io/Eigenvesting-IV-Predicting-Stock-And-Portfolio-Returns-With-Bayesian-Statistics/
'''

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
import pymc3 as pm
import matplotlib.pyplot as plt

# setting the style of the charts
plt.style.use('ggplot')

#Who wants to be a millionaire
np.random.seed(1000000)

#location of the data set
file_location = "data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"

# importing the data set, converting date column to datetime, making the trading date the index for the Pandas DataFrame and sorting the DataFrame by date
df = pd.read_csv(file_location, index_col='date', parse_dates=True)

# creating a DataFrame with just Apple EOD data
AAPL = df.loc[df['ticker'] == "AAPL"][-500:]

# retrieving rows where adj_close is finite
AAPL = AAPL[np.isfinite(AAPL['adj_close'])]

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['adj_close'])
ax.set(title='AAPL', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

# subtract previous value t-1 from the current value t to get the difference d(t) to make the series stationary
AAPL['First Difference'] = AAPL['adj_close'] - AAPL['adj_close'].shift()

# retrieving rows where adj_close is finite
AAPL = AAPL[np.isfinite(AAPL['First Difference'])]

train, test = np.arange(0, 450), np.arange(451, len(AAPL['First Difference']))
n = len(train)

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['First Difference'])
ax.set(title='AAPL', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

with pm.Model() as model:
    sigma = pm.Exponential('sigma', 1. / .02, testval=.1)
    mu = pm.Normal('mu', 0, sd=5, testval=.1)

    nu = pm.Exponential('nu', 1. / 10)
    logs = pm.GaussianRandomWalk('logs', tau=sigma ** -2, shape=n)

    # lam uses variance in pymc3, not sd like in scipy
    r = pm.StudentT('r', nu, mu=mu, lam=1 / np.exp(-2 * logs), observed=AAPL['First Difference'].values[train])

with model:
    start = pm.find_MAP(vars=[logs], fmin=sp.optimize.fmin_l_bfgs_b)

with model:
    step = pm.NUTS(vars=[logs, mu, nu, sigma], scaling=start, gamma=.25)
    start2 = pm.sample(100, step, start=start)[-1]

    # Start next run at the last sampled position.
    step = pm.NUTS(vars=[logs, mu, nu, sigma], scaling=start2, gamma=.55)
    trace = pm.sample(2000, step, start=start2)

pm.traceplot(trace)


def generate_proj_returns(burn_in, trace, len_to_train):
    num_pred = 1000
    mod_returns = np.ones(shape=(num_pred, len_to_train))
    vol = np.ones(shape=(num_pred, len_to_train))
    for k in range(0, num_pred):
        nu = trace[burn_in + k]['nu']
        mu = trace[burn_in + k]['mu']
        sigma = trace[burn_in + k]['sigma']
        s = trace[burn_in + k]['logs'][-1]
        for j in range(0, len_to_train):
            cur_log_return, s = _generate_proj_returns(mu,
                                                       s,
                                                       nu,
                                                       sigma)
            mod_returns[k, j] = cur_log_return
            vol[k, j] = s
    return mod_returns, vol


def _generate_proj_returns(mu, volatility, nu, sig):
    next_vol = np.random.normal(volatility, scale=sig)  # sig is SD

    # Not 1/np.exp(-2*next_vol), scale treated differently in scipy than pymc3
    log_return = stats.t.rvs(nu, mu, scale=np.exp(-1 * next_vol))
    return log_return, next_vol


sim_returns, vol = generate_proj_returns(1000, trace, len(test))

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['First Difference'], color='blue')
ax.plot(1+len(train)+np.arange(0,len(test)), new_ret[1,:], color='red')
ax.set(title='AAPL', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
[ax.plot(1/np.exp(trace[k]['logs']),color='r',alpha=.2) for k in range(1000,len(trace))]
ax.plot(AAPL['First Difference'])
[ax.plot(1+len(train)+np.arange(0,len(test)),1/np.exp(vol[j,:]), alpha=.01, color='y') for j in range(0,1000)]
ax = plt.gca()
ax.set_ylim([-.05,.05])
ax.set(title='AAPL', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

#Convert simulated returns to log-price
prices = np.copy(sim_returns)
for k in range(0, len(prices)):
    cur = np.log(AAPL['adj_close'][test[0]])
    for j in range(0,len(prices[k])):
        cur = cur + prices[k, j]
        prices[k, j] = cur

slope = trace[1000:]['mu'].mean()

trend = np.arange(0, len(test)) * slope

ind = np.arange(len(train) + 1, 1 + len(train) + len(test))
ind2 = np.arange(training_start, 1 + len(train) + len(test))

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
[plt.plot(ind, prices[j, :], alpha=.02, color='r') for j in range(0, 1000)]
plt.plot(ind, trend + np.log(AAPL['adj_close'])[len(train) + 1],
         alpha=1,
         linewidth=2,
         color='black',
         linestyle='dotted')

plt.plot(ind2, np.log(AAPL['adj_close'])[ind2])

ax = plt.gca()
ax.set_ylim([7.4, 7.8])
ax.set_xlim([training_start, len(AAPL['adj_close']) - 2])
fig.tight_layout()