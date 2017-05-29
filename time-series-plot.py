import functions as fc
import numpy as np
from matplotlib import pyplot, mlab


def plot_histogram(y, ticker=''):
    """
    plots a histogram of the stock returns
    :param ticker:
    :param y:
    :return:
    """
    fig = pyplot.figure()

    mu = np.mean(y)  # mean of distribution
    sigma = np.std(y)  # standard deviation of distribution
    x = mu + sigma * np.random.randn(10000)

    num_bins = 50
    # the histogram of the data
    n, bins, patches = pyplot.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    pyplot.plot(bins, y, 'r--')
    pyplot.xlabel('returns')
    pyplot.ylabel('probability')
    pyplot.title('{} Histogram of returns \n $\mu={}$ \n $\sigma={}$'.format(ticker, mu, sigma))

    # Tweak spacing to prevent clipping of ylabel
    pyplot.subplots_adjust(left=0.15)
    fig.tight_layout()
    fig.savefig('charts/{}-histogram.png'.format(ticker))

df = fc.get_time_series('MSFT')
df['log_returns'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
df['log_returns'].dropna(inplace=True)
plot_histogram(y=df['log_returns'], ticker='MSFT')



