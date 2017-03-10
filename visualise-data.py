import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

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

def plot_ticker():
    # creating a DataFrame with just Apple EOD data
    tickers = df.loc[df['ticker'] == "AAPL"]

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