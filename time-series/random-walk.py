import functions as fc
import numpy as np

AAPL = fc.get_time_series('AAPL').asfreq('D', method='ffill').round(2)

fc.plot_end_of_day(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# plotting the histogram of returns
fc.plot_histogram(np.diff(AAPL['adj_close']))

fc.plot_time_series(np.diff(AAPL['adj_close']), lags=30)