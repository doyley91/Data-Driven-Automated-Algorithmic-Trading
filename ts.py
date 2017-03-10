'''
Source: https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

style.use('ggplot')

#location of the data set
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

def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = pd.Series.rolling(timeseries, window=12).mean()
    rolstd = pd.Series.rolling(timeseries, window=12).std()
    rolvar = pd.Series.rolling(timeseries, window=12).var()

    # plotting the adj_close of AAPL
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(timeseries, color='blue', label='Original')
    ax.plot(rolmean, color='red')
    ax.plot(rolstd, color='black')
    ax.plot(rolvar, color='green')
    ax.set(title='AAPL Rolling Mean and Standard Deviation', xlabel='time', ylabel='$')
    ax.legend(['Adjusted Close $', 'Rolling Mean', 'Rolling Standard Deviation', 'Rolling Variance'])
    fig.tight_layout()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

test_stationarity(AAPL['adj_close'])

AAPL_log = np.log(AAPL['adj_close'])

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL_log)
ax.set(title='AAPL', xlabel='time', ylabel='%')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

moving_avg = AAPL_log.rolling(center=False, window=12).mean()

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL_log, color='blue')
ax.plot(moving_avg, color='red')
ax.set(title='AAPL Log Returns and Moving Average', xlabel='time', ylabel='%')
ax.legend(['Log Returns', 'Moving Average'])
fig.tight_layout()

ts_log_moving_avg_diff = AAPL_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)

test_stationarity(ts_log_moving_avg_diff)

expwighted_avg = AAPL_log.ewm(min_periods=0, ignore_na=False, adjust=True, halflife=12).mean()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL_log, color='blue')
ax.plot(expwighted_avg, color='red')
ax.set(title='AAPL Log Returns and Moving Average', xlabel='time', ylabel='%')
ax.legend(['Log Returns', 'Moving Average'])
fig.tight_layout()

ts_log_ewma_diff = AAPL_log - expwighted_avg

test_stationarity(ts_log_ewma_diff)

ts_log_diff = AAPL_log - AAPL_log.shift()

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ts_log_diff)
ax.set(title='AAPL', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

ts_log_diff.dropna(inplace=True)

test_stationarity(ts_log_diff)

decomposition = seasonal_decompose(AAPL_log, freq=36)

decomposition.plot()

ts_log_decompose = decomposition.resid
ts_log_decompose.dropna(inplace=True)

test_stationarity(ts_log_decompose)

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(lag_acf)
ax1.axhline(y=0, linestyle='--', color='gray')
ax1.axhline(y=-1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
ax1.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--', color='gray')
ax1.set(title='Autocorrelation Function', xlabel='time', ylabel='Correlation')
ax1.legend(['Log Returns'])
ax2 = fig.add_subplot(122)
ax2.plot(lag_pacf)
ax2.axhline(y=0, linestyle='--',color='gray')
ax2.axhline(y=-1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
ax2.axhline(y=1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
ax2.set(title='Partial Autocorrelation Function', xlabel='time', ylabel='Correlation')
ax2.legend(['Log Returns'])
fig.tight_layout()