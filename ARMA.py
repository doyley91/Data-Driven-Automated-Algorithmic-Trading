'''
Source: http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/tsa_arma_0.html
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.pyplot import style
from scipy import stats
from statsmodels.graphics.api import qqplot

style.use('ggplot')

# location of the data set
file_location = "data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"

# importing the data set, converting date column to datetime, making the trading date the index for the Pandas DataFrame and sorting the DataFrame by date
df = pd.read_csv(file_location, index_col='date', parse_dates=True)

# creating a DataFrame with just Apple EOD data
AAPL = df.loc[df['ticker'] == "AAPL"][-400:]

# retrieving rows where adj_close is finite
AAPL = AAPL[np.isfinite(AAPL['adj_close'])]

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['adj_close'])
ax.set(title='AAPL', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

fig = plt.figure()
ax1 = fig.add_subplot(211)
# plots the auto-correlation function with lags on the horizontal and the correlations on vertical axis
fig = sm.graphics.tsa.plot_acf(AAPL['adj_close'].values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
# plots the partial auto-correlation function with lags on the horizontal and the correlations on vertical axis
fig = sm.graphics.tsa.plot_pacf(AAPL['adj_close'], lags=40, ax=ax2)
fig.tight_layout()
# saves the plot with a dpi of 300
fig.savefig("charts/AAPL-AC-vs-PAC.png", dpi=300)

arma_mod20 = sm.tsa.ARMA(AAPL['adj_close'], (2, 0)).fit()
print(arma_mod20.params)

arma_mod30 = sm.tsa.ARMA(AAPL['adj_close'], (3, 0)).fit()
print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)

print(arma_mod30.params)

print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)

sm.stats.durbin_watson(arma_mod30.resid.values)

fig = plt.figure()
ax = fig.add_subplot(111)
ax = arma_mod30.resid.plot(ax=ax)
ax.set(title='AAPL', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

resid = arma_mod30.resid

stats.normaltest(resid)

fig = plt.figure()
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)
fig.tight_layout()

fig = plt.figure()
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
fig.tight_layout()

r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1, 41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

predict_adj_close = arma_mod30.predict('2016-06-24', '2017-01-24', dynamic=True)
print(predict_adj_close)

fig, ax = plt.subplots()
ax = AAPL.ix['2016':].plot(ax=ax)
fig = arma_mod30.plot_predict('2016', '2017', dynamic=True, ax=ax, plot_insample=False)


def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()


mean_forecast_err(AAPL['adj_close'], predict_adj_close)
