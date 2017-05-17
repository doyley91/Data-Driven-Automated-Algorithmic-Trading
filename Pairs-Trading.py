'''
Source: https://pymc-devs.github.io/pymc3/notebooks/GLM-rolling-regression.html
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as T
from scipy import optimize

# location of the data set
file_location = "data/WIKI_pairs_212b326a081eacca455e13140d7bb9db.csv"

# importing the data set
df = pd.read_csv(file_location)

# converting date column to datetime
df.date = pd.to_datetime(df.date)

# sorting the DataFrame by date
df.sort_values(by='date')

# making the trading date the index for the Pandas DataFrame
df.set_index('date', inplace=True)

# creating a DataFrame with just Apple EOD data
pairs = df.loc[df['ticker'].isin(["AAPL", "MSFT"])]

# retrieving rows where adj_close is finite
pairs = pairs[np.isfinite(pairs['adj_close'])]

# pivoting the DataFrame to create a column for every ticker
pairs = pairs.pivot(index=None, columns='ticker', values='adj_close')

# plotting the adj_close of AAPL
pairs.plot(figsize=(20, 10))

returns = pairs[-400:].pct_change()

returns.plot(figsize=(20, 10))

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, xlabel='Price AAPL in \$', ylabel='Price MSFT in \$')
colors = np.linspace(0.1, 1, len(pairs))
mymap = plt.get_cmap("winter")
sc = ax.scatter(pairs.AAPL, pairs.MSFT, c=colors, cmap=mymap, lw=0)
cb = plt.colorbar(sc)
cb.ax.set_yticklabels([str(p.date()) for p in pairs[::len(pairs) // 10].index])

with pm.Model() as model_reg:
    pm.glm.glm('AAPL ~ MSFT', pairs)
    trace_reg = pm.sample(2000)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, xlabel='Price MSFT in \$', ylabel='Price AAPL in \$',
                     title='Posterior predictive regression lines')
sc = ax.scatter(pairs.MSFT, pairs.AAPL, c=colors, cmap=mymap, lw=0)
pm.glm.plot_posterior_predictive(trace_reg[100:], samples=100,
                                 label='posterior predictive regression lines',
                                 lm=lambda x, sample: sample['Intercept'] + sample['MSFT'] * x,
                                 eval=np.linspace(pairs.MSFT.min(), pairs.MSFT.max(), 100))
cb = plt.colorbar(sc)
cb.ax.set_yticklabels([str(p.date()) for p in pairs[::len(pairs) // 10].index]);
ax.legend(loc=0)

model_randomwalk = pm.Model()
with model_randomwalk:
    # std of random walk, best sampled in log space.
    sigma_alpha = pm.Exponential('sigma_alpha', 1. / .02, testval=.1)
    sigma_beta = pm.Exponential('sigma_beta', 1. / .02, testval=.1)

# To make the model simpler, we will apply the same coefficient for 50 data points at a time
subsample_alpha = 50
subsample_beta = 50
with model_randomwalk:
    alpha = pm.GaussianRandomWalk('alpha', sigma_alpha ** -2,
                                  shape=len(pairs) // subsample_alpha)
    beta = pm.GaussianRandomWalk('beta', sigma_beta ** -2,
                                 shape=len(pairs) // subsample_beta)

    # Make coefficients have the same length as pairs
    alpha_r = T.repeat(alpha, subsample_alpha)
    beta_r = T.repeat(beta, subsample_beta)

with model_randomwalk:
    # Define regression
    regression = alpha_r + beta_r * pairs.MSFT.values

    # Assume pairs are Normally distributed, the mean comes from the regression.
    sd = pm.Uniform('sd', 0, 20)
    likelihood = pm.Normal('y',
                           mu=regression,
                           sd=sd,
                           observed=pairs.AAPL.values)

with model_randomwalk:
    # First optimize random walk
    start = pm.find_MAP(vars=[alpha, beta], fmin=optimize.fmin_l_bfgs_b)

    # Sample
    step = pm.NUTS(scaling=start)
    trace_rw = pm.sample(2000, step, start=start)

fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(111, xlabel='time', ylabel='alpha', title='Change of alpha over time.')
ax.plot(trace_rw[-1000:][alpha].T, 'r', alpha=.05);
ax.set_xticklabels([str(p.date()) for p in pairs[::len(pairs) // 5].index])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlabel='time', ylabel='beta', title='Change of beta over time')
ax.plot(trace_rw[-1000:][beta].T, 'b', alpha=.05);
ax.set_xticklabels([str(p.date()) for p in pairs[::len(pairs) // 5].index])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlabel='Price MSFT in \$', ylabel='Price AAPL in \$',
                     title='Posterior predictive regression lines')

colors = np.linspace(0.1, 1, len(pairs))
colors_sc = np.linspace(0.1, 1, len(trace_rw[-500::10]['alpha'].T))
mymap = plt.get_cmap('winter')
mymap_sc = plt.get_cmap('winter')

xi = np.linspace(pairs.MSFT.min(), pairs.MSFT.max(), 50)
for i, (alpha, beta) in enumerate(zip(trace_rw[-500::10]['alpha'].T, trace_rw[-500::10]['beta'].T)):
    for a, b in zip(alpha, beta):
        ax.plot(xi, a + b * xi, alpha=.05, lw=1, c=mymap_sc(colors_sc[i]))

sc = ax.scatter(pairs.MSFT, pairs.AAPL, label='data', cmap=mymap, c=colors)
cb = plt.colorbar(sc)
cb.ax.set_yticklabels([str(p.date()) for p in pairs[::len(pairs) // 10].index])
