import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

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

AAPL["regression"] = sm.OLS(AAPL["adj_close"],
                            sm.add_constant(range(len(AAPL.index)),
                                            prepend=True)).fit().fittedvalues

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['adj_close'])
ax.plot(AAPL['regression'])
ax.set(title='AAPL Linear Regresion', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $', 'Linear Regression'])
fig.tight_layout()
fig.savefig('charts/AAPL-linear-regression.png', dpi=300)
