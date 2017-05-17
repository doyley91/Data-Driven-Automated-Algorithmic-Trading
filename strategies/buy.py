import matplotlib.pyplot as plt
import pandas as pd
from zipline.api import order, symbol

stocks = []


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    context.stocks = stocks


def handle_data(context, data):
    """
    Called every minute.
    """
    for stock in context.stocks:
        order(symbol(stock), 100)


def analyze(context=None, results=None):
    # Plot the portfolio and asset data.
    ax1 = plt.subplot(211)
    results.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('Portfolio value (USD)')
    ax2 = plt.subplot(212, sharex=ax1)
    results.AAPL.plot(ax=ax2)
    ax2.set_ylabel('AAPL price (USD)')

    # Show the plot.
    plt.gcf().set_size_inches(18, 8)
    plt.show()


def _test_args():
    """Extra arguments to use when zipline's automated tests run this example.
    """

    return {
        'start': pd.Timestamp('2014-01-01', tz='utc'),
        'end': pd.Timestamp('2014-11-01', tz='utc'),
    }
