'''
Source:
http://www.inertia7.com/projects/time-series-stock-market-python
https://github.com/inertia7/timeSeries_sp500_python/blob/master/scripts.py
'''

import pandas as pd
import numpy as np
import statsmodels.tsa.api as sm
#from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from arch import arch_model

plt.style.use('ggplot')

#location of the data set
file_location = "data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"

# importing the data set, converting date column to datetime, making the trading date the index for the Pandas DataFrame and sorting the DataFrame by date
df = pd.read_csv(file_location, index_col='date', parse_dates=True)

# creating a DataFrame with just Apple EOD data
AAPL = df.loc[df['ticker'] == "AAPL"]

# creating a DataFrame with just Apple EOD data
AAPL_test = df.loc[df['ticker'] == "AAPL"]['2016':'2017']

# creating a DataFrame with just Apple EOD data
AAPL_train = df.loc[df['ticker'] == "AAPL"]['1980':'2015']

# retrieving rows where adj_close is finite
AAPL = AAPL[np.isfinite(AAPL['adj_close'])]

# retrieving rows where adj_close is finite
AAPL_test = AAPL_test[np.isfinite(AAPL_test['adj_close'])]

# retrieving rows where adj_close is finite
AAPL_train = AAPL_train[np.isfinite(AAPL_train['adj_close'])]

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL_train['adj_close'])
ax.set(title='AAPL', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

# DIAGNOSING ACF
acf = plot_acf(AAPL_train['adj_close'], lags=20)

# DIAGNOSING PACF
pacf = plot_pacf(AAPL_train['adj_close'], lags=20)

first_difference = AAPL_train['adj_close'] - AAPL_train['adj_close'].shift().dropna()
#first_difference = first_difference[np.isfinite(first_difference)]

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(first_difference)
ax.set(title='AAPL', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

acfDiff = plot_acf(first_difference, lags=20)

pacfDiff = plot_pacf(first_difference, lags=20)

#AIC must be lowest
mod = sm.ARIMA(AAPL_train['adj_close'], order=(0, 1, 1))

results = mod.fit()

results.summary()

predVals = results.predict(start=len(AAPL_train['adj_close']), end=len(AAPL['adj_close']), typ='levels').dropna()

#predVals = predVals[np.isfinite(predVals)]

forecast = pd.concat([AAPL['adj_close'], predVals], axis=1, keys=['original', 'predicted'])

forecast = forecast[np.isfinite(forecast['predicted'])]

forecast = forecast.dropna()

# plotting the adj_close of AAPL
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(forecast['original'])
ax.plot(forecast['predicted'])
ax.set(title='Actual Vs. Forecasted Values', xlabel='time', ylabel='$')
ax.legend(['Original Adjusted Close $', 'Forecast Adjusted Close $'])
fig.tight_layout()