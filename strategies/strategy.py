import pandas as pd
from zipline.api import order, symbol, record, order_target

csv_data = []


def initialize(context):
    context.security = symbol('AAPL')
    csv_data.append(["date", "MA1", "MA2", "Current Price", "Buy/Sell", "Shares", "PnL", "Cash", "Value"])


def handle_data(context, panel):
    # calculating moving average
    MA1 = panel[context.security].mavg(50)
    MA2 = panel[context.security].mavg(100)
    date = str(panel[context.security].datetime)[:10]

    # calculating price, pnl, portfolio value
    current_price = panel[context.security].price
    current_positions = context.portfolio.positions[symbol('SPY')].amount
    cash = context.portfolio.cash
    value = context.portfolio.portfolio_value
    current_pnl = context.portfolio.pnl

    # to buy stock
    if (MA1 > MA2) and current_positions == 0:
        number_of_shares = int(cash / current_price)
        order(context.security, number_of_shares)
        # recording the data
        record(date=date,
               MA1=MA1,
               MA2=MA2,
               Price=current_price,
               status="buy",
               shares=number_of_shares,
               PnL=current_pnl,
               cash=cash,
               value=value)

        csv_data.append([date,
                         format(MA1, '.2f'),
                         format(MA2, '.2f'),
                         format(current_price, '.2f'),
                         "buy",
                         number_of_shares,
                         format(current_pnl, '.2f'),
                         format(cash, '.2f'),
                         format(value, '.2f')])

    # to sell stocks
    elif (MA1 < MA2) and current_positions != 0:
        order_target(context.security, 0)
        record(date=date,
               MA1=MA1,
               MA2=MA2,
               Price=current_price,
               status="sell",
               shares="--",
               PnL=current_pnl,
               cash=cash,
               value=value)

        csv_data.append([date, format(MA1, '.2f'), format(MA2, '.2f'), format(current_price, '.2f'),
                         "sell", "--", format(current_pnl, '.2f'), format(cash, '.2f'), format(value, '.2f')])

    # do nothing just record the data
    else:
        record(date=date,
               MA1=MA1,
               MA2=MA2,
               Price=current_price,
               status="--",
               shares="--",
               PnL=current_pnl,
               cash=cash,
               value=value)

        csv_data.append([date,
                         format(MA1, '.2f'),
                         format(MA2, '.2f'),
                         format(current_price, '.2f'),
                         "--",
                         "--",
                         format(current_pnl, '.2f'),
                         format(cash, '.2f'),
                         format(value, '.2f')])


def _test_args():
    """Extra arguments to use when zipline's automated tests run this example.
    """

    return {
        'start': pd.Timestamp('1991', tz='utc'),
        'end': pd.Timestamp('2017', tz='utc'),
    }
