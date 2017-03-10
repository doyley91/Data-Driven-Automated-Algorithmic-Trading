import functions as fc
import numpy as np
import matplotlib.pyplot as plt

AAPL = fc.return_ticker('AAPL')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(AAPL['adj_close'])
ax.set(title='AAPL', xlabel='time', ylabel='$')
ax.legend(['Adjusted Close $'])
fig.tight_layout()

fc.tsplot(np.diff(AAPL['adj_close']), lags=30)