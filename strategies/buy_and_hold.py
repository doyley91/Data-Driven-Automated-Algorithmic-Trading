import pandas as pd
from zipline.api import order, symbol

stocks = []


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    context.has_ordered = False
    context.stocks = stocks


def handle_data(context, panel):
    """
    Called every minute.
    """
    if not context.has_ordered:
        for stock in context.stocks:
            order(symbol(stock), 100)
        context.has_ordered = True


def _test_args():
    """Extra arguments to use when zipline's automated tests run this example.
    """
    return {
        'start': pd.Timestamp('2008', tz='utc'),
        'end': pd.Timestamp('2013', tz='utc'),
    }
