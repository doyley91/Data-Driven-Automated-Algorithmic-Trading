'''
Source: https://mrjbq7.github.io/ta-lib/func_groups/price_transform.html
'''

import numpy as np
import pandas as pd
import talib
from talib import MA_Type

# location of the data set
file_location = "data/WIKI_PRICES.csv"

# importing the data set
df = pd.read_csv(file_location)

# converting date column to datetime
df.date = pd.to_datetime(df.date)

# sorting the DataFrame by date
df.sort_values(by='date')

# making the trading date the index for the Pandas DataFrame
df.set_index('date', inplace=True)

# pivoting the DataFrame to create a column for every ticker
pdf = df.pivot(index=None, columns='ticker', values='adj_close')

# calculate a simple moving average of the close prices
df['sma_adj_close'] = pdf.apply(talib.SMA(np.array(pdf.values), 5))

# calculating bollinger bands, with triple exponential moving average
df['upper'], df['middle'], df['lower'] = df.apply(talib.BBANDS(np.array(df['adj_close']), matype=MA_Type.T3))

# calculating momentum of the close prices, with a time period of 5
df['mom_adj_close'] = df.apply(talib.MOM(np.array(df['adj_close']), timeperiod=5))

# AD - Chaikin A/D Line
df['real'] = df.apply(talib.AD(df['adj_high'], df['adj_low'], df['adj_close'], df['adj_volume']))

# ADOSC - Chaikin A/D Oscillator
df['real'] = df.apply(
    talib.ADOSC(df['adj_high'], df['adj_low'], df['adj_close'], df['adj_volume'], fastperiod=3, slowperiod=10))

# OBV - On Balance Volume
df['real'] = df.apply(talib.OBV(df['adj_close'], df['adj_volume']))

df['sma_50'] = [talib.SMA(np.array(x.index(), dtype='float'), 20) for x in df.iteritems()]

[print(x) for x in df['adj_close'].iteritems()]

[print(talib.SMA(np.asarray(x), 20)) for x in df['ticker'].iteritems()]

# creating a DataFrame with just Apple EOD data
AAPL = df.loc[df['ticker'] == "AAPL"]
AAPL = AAPL.loc['2000-6-1':'2017-6-1']

AAPL['SMA_20'] = talib.SMA(np.asarray(AAPL['adj_close']), 20)
AAPL['SMA_50'] = talib.SMA(np.asarray(AAPL['adj_close']), 50)
AAPL.plot(y=['adj_close', 'SMA_20', 'SMA_50'], title='AAPL Close & Moving Averages')

AAPL['upper'], AAPL['middle'], AAPL['lower'] = talib.BBANDS(np.array(AAPL['adj_close']), matype=MA_Type.T3)
AAPL.plot(y=['adj_close', 'upper', 'middle', 'lower'], title='AAPL triple exponential bollinger bands')

AAPL['momentum'] = talib.MOM(np.array(AAPL['adj_close']), timeperiod=5)
AAPL.plot(y=['adj_close', 'momentum'], title='AAPL momentum')
