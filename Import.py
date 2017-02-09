#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:23:06 2017

@author: gabegm
"""

import datetime as dt
import pandas as pd
import numpy as np
from scipy import stats, integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
style.use('ggplot')

file_location = "data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"

#tickers to use
tickers_to_plot = ['AAPL', 'GOOG', 'TSLA', 'YHOO', 'QCOM', 'NFLX', 'MFST', 'INTC', 'HPQ']

#nitpicking specific features in the DataFrame
useful_feats_to_plot = ['open', 'adj_open']

#start date
start = dt(2014, 1, 1)

#end date
end = dt(2014, 3, 28)

#importing the data set
df = pd.read_csv(file_location)

#printing the numbers of rows and columns
df.shape

#describes the index
df.index

#printing the columns of the DataFrame
df.columns

#printing the info of the DataFrame
df.info()

#printing the number of non-NaN values per column
df.count()

#printing the number of NaN values per column
len(df) - df.count()

#printing the sum of values of the DataFrame
df.sum()

#printing the cumulative sum of values
df.cumsum()

#printing the minimum/maximum values
df.min()/df.max()

#printing the minimum/maximum index values
df.idmin()/df.idmax()

#printing summary statistics
df.describe()

#printing mean of values
df.mean()

#printing median of values
df.median()

#printing the first 5 lines
df.head()

#printing the last 5 lines
df.tail()

#converting date column to datetime
df.date = pd.to_datetime(df.date)

#sorting the DataFrame by date
df = df.sort_values(by='date')

#making the trading date the index for the Pandas DataFrame
df.set_index('date', inplace=True)

#drop only if NaN in specific column
df.dropna(subset=[1])

#drop row if it does not have at least two values that are **not** NaN
df.dropna(thresh=2)

#drop only if ALL columns are NaN
df.dropna(how='all')

#dropping all NaN values
cdf = df.dropna()

#printing all rows from 2006
cdf.loc['2006'].head()

#creating a DataFrame with the correlation values of every column to every column
df_corr = df.pivot(index=None, columns='ticker', values='adj_close').corr()

#creating an array of the values of correlations in the DataFrame
data1 = df_corr.values

#creating a new figure
fig1 = plt.figure()

#creating an axis
ax1 = fig1.add_subplot(111)

#creating a heatmap with colours going from red (negative correlations) to yellow (no correlations) to green (positive correlations)
heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)

#creating a colour side bar as a scale for the heatmap
fig1.colorbar(heatmap1)

#setting the ticks of the x-axis
ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)

#setting the ticks of the y-axis
ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)

#inverts the scale of the y-axis
ax1.invert_yaxis()

#places the x-axis at the top of the graph
ax1.xaxis.tick_top()

#storing the ticker labels in an array
column_labels = df_corr.columns

#storing the dates in an array
row_labels = df_corr.index

#setting the x-axis labels to the dates
ax1.set_xticklabels(column_labels)

#setting the y-axis labels to the ticker labels
ax1.set_yticklabels(row_labels)

#rotates the x-axis labels vertically to fit the graph
plt.xticks(rotation=90)

#sets the range from -1 to 1
heatmap1.set_clim(-1, 1)

#automatically adjusts subplot paramaters to give specified padding
plt.tight_layout()

#shows the plot
plt.show()

#saves the plot with a dpi of 300
plt.savefig("correlations.png", dpi=300)

#creating a DataFrame with just Apple EOD data
tickers = df.loc[df['ticker'] == "AAPL"]

#creating a DataFrame with Apple and Google EOD data
tickers = df.loc[df['ticker'].isin(tickers_to_plot)]

#plotting the closing price over the entire date range in the DataFrame
tickers['close'].plot(kind='line', figsize=(16, 12), title="AAPL", legend=True)

#creating a column in the DataFrame to calculate the 100 day moving average on the adjusted close price
tickers['100ma'] = tickers['adj_close'].rolling(window=100, min_periods=0).mean()

#creating a subplot with a 6x1 grid and starts at (0,0)
ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)

#creating a subplot with a 6x1 grid, starts at (5,0) and aligns its x-axis with ax1
ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)

#plotting the line graph for the adjusted close price
ax1.plot(tickers.index, tickers['adj_close'])

#plotting the line graph for the 100 day moving average
ax1.plot(tickers.index, tickers['100ma'])

#plotting a bar chart of the adj_volume
ax2.bar(tickers.index, tickers['adj_volume'])

#plotting the graph
plt.show()

#saving plot to disk
plt.savefig("my_plot2.png")

#closes the plot
plt.close()

#creating a new DataFrame based on the adjusted_close price resampled with a 10 day window
tickers_ohlc = tickers['adj_close'].resample('10D').ohlc()

#creating a new DataFrame based on the adjusted volume resampled with a 10 day window
tickers_volume = tickers['adj_volume'].resample('10D').sum()

#resetting the index of the DataFrame
tickers_ohlc = tickers_ohlc.reset_index()

#converting the date column to mdates
tickers_ohlc['date'] = tickers_ohlc['date'].map(mdates.date2num)

#creating a new figure
fig = plt.figure(figsize=(16, 12))

#creating a subplot with a 6x1 grid and starts at (0,0)
ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)

#creating a subplot with a 6x1 grid and starts at (5,0)
ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)

#converts the axis from raw mdate numbers to dates
ax1.xaxis_date()

#plotting the candlestick graph
candlestick_ohlc(ax1, tickers_ohlc.values, width=2, colorup='g')

#plotting the volume bar chart
ax2.fill_between(tickers_volume.index.map(mdates.date2num), tickers_volume.values, 0)

df = pd.DataFrame(pd.date_range('1/1/2016', periods=10, freq='D'), columns=['date'])
