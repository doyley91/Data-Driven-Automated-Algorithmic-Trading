'''
Source:
http://www.johnwittenauer.net/a-simple-time-series-analysis-of-the-sp-500-index/
http://nbviewer.jupyter.org/github/jdwittenauer/ipython-notebooks/blob/master/notebooks/misc/TimeSeriesStockAnalysis.ipynb
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

plt.style.use('ggplot')

#location of the data set
file_location = "data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"

# importing the data set, converting date column to datetime, making the trading date the index for the Pandas DataFrame and sorting the DataFrame by date
df = pd.read_csv(file_location, index_col='date', parse_dates=True)

# creating a DataFrame with just Apple EOD data
AAPL = df.loc[df['ticker'] == "AAPL"]['1998':'2015']

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

# plotting the first difference
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['First Difference'])
ax.set(title='AAPL First Diference', xlabel='time', ylabel='$')
ax.legend(['First Difference'])
fig.tight_layout()
# saves the plot with a dpi of 300
fig.savefig("charts/AAPL-first-difference.png", dpi=300)

# applying a log transform to the original adj_close to flatten the data from an exponential curve to a linear curve
AAPL['Natural Log'] = AAPL['adj_close'].apply(lambda x: np.log(x))

# plotting the natural log
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['Natural Log'])
ax.set(title='AAPL Natural Log', xlabel='time', ylabel='$')
ax.legend(['Natural Log'])
fig.tight_layout()
# saves the plot with a dpi of 300
fig.savefig("charts/AAPL-natural-log.png", dpi=300)

# rolling variance statistic with original time series
AAPL['Original Variance'] = pd.Series.rolling(AAPL['adj_close'],
                                              window=30,
                                              min_periods=None,
                                              freq=None,
                                              center=True).var()

# rolling variance statistic with logged time series
AAPL['Log Variance'] = pd.Series.rolling(AAPL['Natural Log'],
                                         window=30,
                                         min_periods=None,
                                         freq=None,
                                         center=True).var()

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(AAPL['Original Variance'])
ax1.set(title='AAPL Original Variance', xlabel='time', ylabel='$')
ax1.legend(['Original Variance'])
ax2 = fig.add_subplot(212)
ax2.plot(AAPL['Log Variance'])
ax2.set(title='AAPL Log Variance', xlabel='time', ylabel='$')
ax2.legend(['Log Variance'])
# fixes overlapping in layout
fig.tight_layout()
# saves the plot with a dpi of 300
fig.savefig("charts/AAPL-original-vs-log-variance.png", dpi=300)

# taking the first difference of the logged time series
AAPL['Logged First Difference'] = AAPL['Natural Log'] - AAPL['Natural Log'].shift()

# plotting the logged first difference
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['Logged First Difference'])
ax.set(title='AAPL Logged First Difference', xlabel='time', ylabel='$')
ax.legend(['Logged First Difference'])
fig.tight_layout()
# saves the plot with a dpi of 300
fig.savefig("charts/AAPL-logged-first-difference.png", dpi=300)

# creating a 1 day lagged version of the time series to check for seasonality
AAPL['Lag 1'] = AAPL['Logged First Difference'].shift()

# creating a 2 day lagged version of the time series to check for seasonality
AAPL['Lag 2'] = AAPL['Logged First Difference'].shift(2)

# creating a 5 day lagged version of the time series to check for seasonality
AAPL['Lag 5'] = AAPL['Logged First Difference'].shift(5)

# creating a 30 day lagged version of the time series to check for seasonality
AAPL['Lag 30'] = AAPL['Logged First Difference'].shift(30)

# plotting the relationship of the 1 day lagged variables with a scatterplot of original variable vs lagged variable
sns.jointplot('Logged First Difference', 'Lag 1', AAPL, kind='reg', size=13)

# saving the plot to memory
sns_plot = sns.jointplot('Logged First Difference', 'Lag 1', AAPL, kind='reg', size=13)

# saves the plot with a dpi of 300
sns_plot.savefig("charts/AAPL-logged-first-difference-lagged-1.png", dpi=300)

# plotting the relationship of the 2 day lagged variables with a scatterplot of original variable vs lagged variable
sns.jointplot('Logged First Difference', 'Lag 2', AAPL, kind='reg', size=13)

# saving the plot to memory
sns_plot = sns.jointplot('Logged First Difference', 'Lag 2', AAPL, kind='reg', size=13)

# saves the plot with a dpi of 300
sns_plot.savefig("charts/AAPL-logged-first-difference-lagged-2.png", dpi=300)

# plotting the relationship of the 5 day lagged variables with a scatterplot of original variable vs lagged variable
sns.jointplot('Logged First Difference', 'Lag 5', AAPL, kind='reg', size=13)

# saving the plot to memory
sns_plot = sns.jointplot('Logged First Difference', 'Lag 5', AAPL, kind='reg', size=13)

# saves the plot with a dpi of 300
sns_plot.savefig("charts/AAPL-logged-first-difference-lagged-5.png", dpi=300)

# plotting the relationship of the 30 day lagged variables with a scatterplot of original variable vs lagged variable
sns.jointplot('Logged First Difference', 'Lag 30', AAPL, kind='reg', size=13)

# saving the plot to memory
sns_plot = sns.jointplot('Logged First Difference', 'Lag 30', AAPL, kind='reg', size=13)

# saves the plot with a dpi of 300
sns_plot.savefig("charts/AAPL-logged-first-difference-lagged-30.png", dpi=300)

# checking for auto correlation
lag_correlations = acf(AAPL['Logged First Difference'].iloc[1:])

# checking for partial auto correlation
lag_partial_correlations = pacf(AAPL['Logged First Difference'].iloc[1:])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lag_correlations, marker='o', linestyle='--')
ax.plot(lag_partial_correlations, marker='v', linestyle=':')
ax.set(title='AAPL Lagged Correlations vs. Partial Lagged Correlations', xlabel='time', ylabel='Correlation')
ax.legend(['Lagged Correlations', 'Partial Lagged Correlations'])
fig.tight_layout()
# saves the plot with a dpi of 300
fig.savefig("charts/AAPL-autocorrelation.png", dpi=300)

# breaking down the time series into a seasonal factor
decomposition = seasonal_decompose(AAPL['Natural Log'], model='additive', freq=30)

# plotting the decomposition
decomposition.plot()

# fitting an ARIMA model to the natural log time series
model = sm.tsa.ARIMA(AAPL['Natural Log'].iloc[1:], order=(1, 0, 0))

# saving the forecast values
results = model.fit(disp=-1)

# saving the forecast values in a new column
AAPL['Forecast'] = results.fittedvalues

# plotting the natural log against the forecast
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['Natural Log'])
ax.plot(AAPL['Forecast'])
ax.set(title='AAPL Natural Log vs Forecast', xlabel='time', ylabel='$')
ax.legend(['Natural Log', 'Forecast'])
# saves the plot with a dpi of 300
fig.savefig("charts/AAPL-natural-log-vs-forecast.png", dpi=300)

# fitting an ARIMA model to the logged first difference time series
model = sm.tsa.ARIMA(AAPL['Logged First Difference'].iloc[1:], order=(1, 0, 0))

# saving the forecast values
results = model.fit(disp=-1)

# saving the forecast values in a new column
AAPL['Forecast'] = results.fittedvalues

# plotting the natural log against the forecast
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['Logged First Difference'])
ax.plot(AAPL['Forecast'])
ax.set(title='AAPL Natural Log vs Forecast', xlabel='time', ylabel='$')
ax.legend(['Natural Log', 'Forecast'])
# saves the plot with a dpi of 300
fig.savefig("charts/AAPL-logged-first-difference-vs-forecast.png", dpi=300)

# fitting an exponential smoothing ARIMA model to the logged first difference time series
model = sm.tsa.ARIMA(AAPL['Logged First Difference'].iloc[1:], order=(0, 0, 1))

# saving the forecast values
results = model.fit(disp=-1)

# saving the forecast values in a new column
AAPL['Forecast'] = results.fittedvalues

# plotting the natural log against the forecast
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['Logged First Difference'])
ax.plot(AAPL['Forecast'])
ax.set(title='AAPL Logged First Difference vs Forecast', xlabel='time', ylabel='$')
ax.legend(['Logged First Difference', 'Forecast'])
# saves the plot with a dpi of 300
fig.savefig("charts/AAPL-logged-first-difference-vs-forecast.png", dpi=300)