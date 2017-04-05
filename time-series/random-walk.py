import functions as fc
import numpy as np

AAPL = fc.get_time_series('AAPL').asfreq('D', method='ffill')

fc.end_of_day_plot(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

fc.tsplot(np.diff(AAPL['adj_close']), lags=30)