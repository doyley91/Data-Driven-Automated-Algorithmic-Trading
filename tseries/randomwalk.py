import functions as fc
import numpy as np

df = fc.get_time_series('AAPL')

fc.plot_end_of_day(df['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

df['first_difference'] = df['adj_close'].diff()

df['first_difference'].dropna(inplace=True)

# plotting the histogram of returns
fc.plot_histogram(np.diff(df['first_difference']))

fc.plot_time_series(np.diff(df['first_difference']), lags=30)
