'''
Source: https://pymc-devs.github.io/pymc3/notebooks/stochastic_volatility.html
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.distributions.timeseries import GaussianRandomWalk

# not quite sure what this does but we need it
n = 400

# location of the data set
file_location = "data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"

# importing the data set
df = pd.read_csv(file_location)

# converting date column to datetime
df.date = pd.to_datetime(df.date)

# sorting the DataFrame by date
df.sort_values(by='date')

# making the trading date the index for the Pandas DataFrame
df.set_index('date', inplace=True)

# creating a DataFrame with just Apple EOD data
AAPL = df.loc[df['ticker'] == "AAPL"]

# retrieving rows where adj_close is finite
AAPL = AAPL[np.isfinite(AAPL['adj_close'])]

# plotting the adj_close of AAPL
AAPL['adj_close'].plot(figsize=(20, 10))

# subtract previous value t-1 from the current value t to get the difference d(t) to make the series stationary
AAPL['First Difference'] = AAPL['adj_close'] - AAPL['adj_close'].shift()

# plotting the natural log
AAPL['First Difference'].plot(figsize=(20, 10))

# storing the variance in a Series; not sure why we need the -n: but it seems to remove the first couple of values
variance = AAPL['First Difference'][-n:]

# dropping the date index
variance = variance.reset_index(drop=True)

# converting the Pandas Series to a Numpy array
variance = variance.as_matrix()

plt.plot(variance)

model = pm.Model()
with model:
    sigma = pm.Exponential('sigma', 1. / .02, testval=.1)

    nu = pm.Exponential('nu', 1. / 10)
    s = GaussianRandomWalk('s', sigma ** -2, shape=n)

    r = pm.StudentT('r', nu, lam=pm.math.exp(-2 * s), observed=variance)

with model:
    trace = pm.sample(2000)

pm.traceplot(trace, model.vars[:-1], figsize=(20, 10))

# creating a new figure
fig = plt.figure(figsize=(16, 12))

# saving the plot to memory
ax = plt.plot(trace[s][::10].T, 'b', alpha=.03)

# plot title
fig.suptitle(str(s), fontsize=20)

# x-label
plt.xlabel('time', fontsize=18)

# y-label
plt.ylabel('log volatility', fontsize=16)

# saves the plot with a dpi of 300
fig.savefig("charts/AAPL-natural-log.png", dpi=300)

# closes the plot
plt.close()

# creating a new figure
fig = plt.figure(figsize=(16, 12))

ax1 = plt.plot(np.abs(variance))

# ax1.set_xlim(xmin=0)

ax2 = plt.plot(np.exp(trace[s][::10].T), 'r', alpha=.03)

# ax2.set_xlim(xmin=0)

fig.suptitle(str(s), fontsize=20)

plt.xlabel('time', fontsize=18)

plt.ylabel('absolute returns', fontsize=16)

# saves the plot with a dpi of 300
fig.savefig("charts/AAPL-natural-log.png", dpi=300)

plt.close()
