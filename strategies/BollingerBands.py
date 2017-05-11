import functions as fc
import pandas as pd
from collections import OrderedDict
import locale as loc
from zipline.algorithm import TradingAlgorithm
from pytz import timezone
from dateutil.tz import tzutc
from dateutil.parser import parse
from zipline.finance.slippage import FixedSlippage
from zipline.finance.commission import PerTrade

central = timezone('US/Central')
HOLDTIME = 5
loc.setlocale(loc.LC_ALL, 'en_US.UTF-8')
COMMISSION = 0.005


def date_utc(s):
    return parse(s, tzinfos=tzutc)


class BollingerBands(TradingAlgorithm):
    def initialize(self):
        """
        Called once at the start of the algorithm.
        """
        self.invested = False

        self.trade_size = 1000
        self.long = False
        self.short = False
        self.closed = False
        self.trading_day_counter = 0
        self.pct_stop = 0.025
        self.long_stop_price = 0.0
        self.short_stop_price = 0.0
        self.target = 0.05
        commission_cost = self.trade_size * COMMISSION
        self.set_slippage(FixedSlippage(spread=0.10))
        self.set_commission(PerTrade(cost=commission_cost))

    def handle_data(self, data):
        """
        Called every minute.
        """
        for ticker, df in data.items():
            self.order(df['adj_close'], self.trade_size)
            self.long_stop_price = df['adj_open'] - df['adj_open'] * float(self.pct_stop)
            self.short_stop_price = df['adj_open'] + df['adj_open'] * float(self.target)
            self.long = True
            self.closed = False
            self.invested = True
            self.trading_day_counter = 0

            self.trading_day_counter = self.trading_day_counter + 1
            self.record(Close=df['adj_close'],
                        upper=df['upper'],
                        lower=df['lower'],
                        long=self.long,
                        short=self.short,
                        holdtime=self.trading_day_counter,
                        closed_position=self.closed,
                        shares=self.trade_size)


if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT']

    data = OrderedDict()

    for ticker in tickers:
        data[ticker] = fc.get_time_series(ticker)
        data[ticker]['avg'] = data[ticker]['adj_close'].rolling(window=21, center=False).mean()
        data[ticker]['std'] = data[ticker]['adj_close'].rolling(window=21, center=False).std()
        data[ticker]['upper'] = data[ticker]['avg'] + 2 * data[ticker]['std']
        data[ticker]['lower'] = data[ticker]['avg'] - 2 * data[ticker]['std']
        data[ticker] = data[ticker].dropna()

    # converting dataframe data into panel
    panel = pd.Panel(data)
    panel.minor_axis = ['ticker',
                        'open',
                        'high',
                        'low',
                        'close',
                        'volume',
                        'ex-dividend',
                        'split_ratio',
                        'adj_open',
                        'adj_high',
                        'adj_low',
                        'adj_close',
                        'adj_volume',
                        'avg',
                        'std',
                        'upper',
                        'lower']

    # # # # init Strat Class
    Strategy = BollingerBands()
    # #print df

    # # # # # # Run Strategy
    results = Strategy.run(panel)
    results['algorithm_returns'] = (1 + results.returns).cumprod()

    results.to_csv('output.csv')
    print(results['algorithm_returns'].tail(1)[0] * 100)
