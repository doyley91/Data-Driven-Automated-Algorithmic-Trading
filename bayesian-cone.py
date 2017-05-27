'''
Source: https://blog.quantopian.com/bayesian-cone/
'''

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pymc3 as pm
from pymc3.distributions.timeseries import GaussianRandomWalk

# setting the style of the charts
plt.style.use('ggplot')

# location of the data set
file_location = "data/WIKI_PRICES.csv"

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

returns = AAPL['adj_close'][-400:].pct_change()

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(returns)
ax.set(title='AAPL', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

with pm.Model():
    mu = pm.Normal('mean returns', mu=0, sd=.01, testval=returns.mean())
    sigma = pm.HalfCauchy('volatility', beta=1, testval=returns.std())
    returns = pm.Normal('returns', mu=mu, sd=sigma, observed=returns)

    # Fit the model
    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    trace = pm.sample(2000, step, start=start)

pm.traceplot(trace)

with pm.Model():
    mu = pm.Normal('mean returns', mu=0, sd=.01)
    sigma = pm.HalfCauchy('volatility', beta=1)
    nu = pm.Exponential('nu_minus_two', 1. / 10.)

    returns = pm.T('returns', nu=nu + 2, mu=mu, sd=sigma, observed=returns)

    # Fit model to data
    start = pm.find_MAP(fmin=sp.optimize.fmin_powell)
    step = pm.NUTS(scaling=start)
    trace = pm.sample(2000, step, start=start)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(returns)
ax.plot(returns.index, 1 / np.exp(trace['s', ::5].T), 'r', alpha=.03)
ax.set(title='volatility_process', xlabel='time', ylabel='volatility')
ax.legend(['S&P500', 'stochastic volatility process'])

# saves the plot with a dpi of 300
fig.savefig("charts/No-U-Turn-Sampler.png", dpi=300)
