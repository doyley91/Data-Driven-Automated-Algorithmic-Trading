import functions as fc
import numpy as np

AAPL = fc.get_time_series('AAPL')

fc.plot_end_of_day(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

AAPL['first_difference'] = AAPL['adj_close'].diff()

AAPL['first_difference'].dropna(inplace=True)

# plotting the histogram of returns
fc.plot_histogram(np.diff(AAPL['first_difference']))

fc.plot_time_series(np.diff(AAPL['first_difference']), lags=30)