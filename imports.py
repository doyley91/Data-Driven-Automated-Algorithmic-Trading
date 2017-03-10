#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:23:06 2017

@author: gabegm
"""

import pandas as pd

#location of the data set
file_location = "data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"

# importing the data set
df = pd.read_csv(file_location)

# converting date column to datetime
df.date = pd.to_datetime(df.date)

# sorting the DataFrame by date
df.sort_values(by='date')

# making the trading date the index for the Pandas DataFrame
df.set_index('date', inplace=True)

# printing the info of the DataFrame
df.info()

#nitpicking specific features in the DataFrame
features = ['open', 'adj_open']

#start date
start = dt(2014, 1, 1)

#end date
end = dt(2014, 3, 28)

#printing the numbers of rows and columns
df.shape

#describes the index
df.index

#printing the columns of the DataFrame
df.columns

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

#df = pd.DataFrame(pd.date_range('1/1/2016', periods=10, freq='D'), columns=['date'])