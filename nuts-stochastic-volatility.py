'''
Source: https://pymc-devs.github.io/pymc3/notebooks/getting_started.html
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3 import Exponential, StudentT, Deterministic
from pymc3.distributions.timeseries import GaussianRandomWalk
from pymc3.math import exp

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

with pm.Model() as sp500_model:
    nu = Exponential('nu', 1. / 10, testval=5.)
    sigma = Exponential('sigma', 1. / .02, testval=.1)
    s = GaussianRandomWalk('s', sigma ** -2, shape=len(returns))
    volatility_process = Deterministic('volatility_process', exp(-2 * s))
    r = StudentT('r', nu, lam=1 / volatility_process, observed=returns)

with sp500_model:
    trace = pm.sample(2000)

pm.traceplot(trace[200:], [nu, sigma])

fig, ax = plt.subplots()
returns.plot(ax=ax)
ax.plot(returns.index, 1 / np.exp(trace['s', ::5].T), 'r', alpha=.03)
ax.set(title='volatility_process', xlabel='time', ylabel='volatility')
ax.legend(['S&P500', 'stochastic volatility process'])

# saves the plot with a dpi of 300
fig.savefig("charts/No-U-Turn-Sampler.png", dpi=300)
