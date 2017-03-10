'''
Source: https://github.com/statsmodels/statsmodels/blob/master/statsmodels/sandbox/examples/example_garch.py
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import scikits.timeseries as ts
#import scikits.timeseries.lib.plotlib as tpl
import statsmodels.api as sm
from statsmodels.sandbox import tsa

#location of the data set
file_location = "data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"

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

cl = AAPL['adj_close']
ret = np.diff(np.log(cl))[-2000:]*1000.
ggmod = Garch(ret - ret.mean())#hgjr4[:nobs])#-hgjr4.mean()) #errgjr4)
ggmod.nar = 1
ggmod.nma = 1
ggmod._start_params = np.array([-0.1, 0.1, 0.1, 0.1])
ggres = ggmod.fit(start_params=np.array([-0.1, 0.1, 0.1, 0.0]),
                  maxiter=1000,method='bfgs')
print('ggres.params', ggres.params)
garchplot(ggmod.errorsest, ggmod.h, title='Garch estimated')

ggmod0 = Garch0(ret - ret.mean())#hgjr4[:nobs])#-hgjr4.mean()) #errgjr4)
ggmod0.nar = 1
ggmod.nma = 1
start_params = np.array([-0.1, 0.1, ret.var()])
ggmod0._start_params = start_params #np.array([-0.6, 0.1, 0.2, 0.0])
ggres0 = ggmod0.fit(start_params=start_params, maxiter=2000)
print('ggres0.params', ggres0.params)

g11res = optimize.fmin(lambda params: -loglike_GARCH11(params, ret - ret.mean())[0], [0.01, 0.1, 0.1])
print(g11res)
llf = loglike_GARCH11(g11res, ret - ret.mean())
print(llf[0])

ggmod0 = Garch0(ret - ret.mean())#hgjr4[:nobs])#-hgjr4.mean()) #errgjr4)
ggmod0.nar = 2
ggmod.nma = 2
start_params = np.array([-0.1,-0.1, 0.1, 0.1, ret.var()])
ggmod0._start_params = start_params #np.array([-0.6, 0.1, 0.2, 0.0])
ggres0 = ggmod0.fit(start_params=start_params, maxiter=2000)#, method='ncg')
print('ggres0.params', ggres0.params)

ggmod = Garch(ret - ret.mean())#hgjr4[:nobs])#-hgjr4.mean()) #errgjr4)
ggmod.nar = 2
ggmod.nma = 2
start_params = np.array([-0.1,-0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
ggmod._start_params = start_params
ggres = ggmod.fit(start_params=start_params, maxiter=1000)#,method='bfgs')
print('ggres.params', ggres.params)