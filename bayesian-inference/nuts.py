'''
Source: https://pymc-devs.github.io/pymc3/notebooks/getting_started.html
'''

import functions as fc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from pymc3 import Exponential, StudentT, Deterministic
from pymc3.math import exp
from pymc3.distributions.timeseries import GaussianRandomWalk

# setting the style of the charts
plt.style.use('ggplot')

AAPL = fc.return_ticker('AAPL').asfreq('D', method='ffill')

fc.eodplot(AAPL)

training_set = AAPL[:-500]
test_set = AAPL[-500:]

returns = training_set['adj_close'].pct_change()

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(returns)
ax.set(title='AAPL', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

with pm.Model() as sp500_model:
    nu = Exponential('nu', 1./10, testval=5.)
    sigma = Exponential('sigma', 1./.02, testval=.1)
    s = GaussianRandomWalk('s', sigma**-2, shape=len(returns))
    volatility_process = Deterministic('volatility_process', exp(-2*s))
    r = StudentT('r', nu, lam=1/volatility_process, observed=returns)

with sp500_model:
    trace = pm.sample(2000)

pm.traceplot(trace[200:], [nu, sigma])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(returns)
ax.plot(returns.index, 1/np.exp(trace['s', ::5].T), 'r', alpha=.03)
ax.set(title='volatility_process', xlabel='time', ylabel='volatility')
ax.legend(['S&P500', 'stochastic volatility process'])
fig.tight_layout()
fig.savefig("charts/No-U-Turn-Sampler.png", dpi=300)