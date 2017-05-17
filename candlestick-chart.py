import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from mpl_finance import candlestick_ohlc

# location of the data set
file_location = "data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"

# importing the data set, converting date column to datetime, making the trading date the index for the Pandas DataFrame and sorting the DataFrame by date
df = pd.read_csv(file_location, index_col='date', parse_dates=True)

# tickers to use
portfolio = ['AAPL', 'GOOG', 'TSLA', 'YHOO', 'QCOM', 'NFLX', 'MFST', 'INTC', 'HPQ']

# creating a DataFrame with Apple and Google EOD data
tickers = df.loc[df['ticker'].isin(portfolio)]

# creating a new DataFrame based on the adjusted_close price resampled with a 10 day window
tickers_ohlc = tickers['adj_close'].resample('10D').ohlc()

# creating a new DataFrame based on the adjusted volume resampled with a 10 day window
tickers_volume = tickers['adj_volume'].resample('10D').sum()

# resetting the index of the DataFrame
tickers_ohlc = tickers_ohlc.reset_index()

# converting the date column to mdates
tickers_ohlc['date'] = tickers_ohlc['date'].map(mdates.date2num)

# creating a new figure
fig = plt.figure()

# creating a subplot with a 6x1 grid and starts at (0,0)
ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)

# creating a subplot with a 6x1 grid and starts at (5,0)
ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)

# converts the axis from raw mdate numbers to dates
ax1.xaxis_date()

# plotting the candlestick graph
candlestick_ohlc(ax1, tickers_ohlc.values, width=2, colorup='g')

# plotting the volume bar chart
ax2.fill_between(tickers_volume.index.map(mdates.date2num), tickers_volume.values, 0)

fig.tight_layout()

# saves the plot with a dpi of 300
fig.savefig("charts/candlestick-chart.png", dpi=300)
