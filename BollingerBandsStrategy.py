import locale

from dateutil.parser import parse
from dateutil.tz import tzutc
from pytz import timezone
from zipline.algorithm import TradingAlgorithm
from zipline.finance.commission import PerTrade
from zipline.finance.slippage import FixedSlippage

import functions as fc

central = timezone('US/Central')
HOLDTIME = 5
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
COMMISSION = 0.005


def date_utc(s):
    return parse(s, tzinfos=tzutc)


class BollingerBands(TradingAlgorithm):
    def initialize(self):
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

        if data['adj_close'].price >= data['upper'].price and not self.invested:
            self.order('adj_close', self.trade_size)
            self.long_stop_price = data['Open'].price - data['Open'].price * float(self.pct_stop)
            self.short_stop_price = data['Open'].price + data['Open'].price * float(self.target)
            self.long = True
            self.closed = False
            self.invested = True
            self.trading_day_counter = 0
        if data['adj_close'].price <= data['lower'].price and not self.invested:
            self.short_stop_price = data['Open'].price + data['Open'].price * float(self.pct_stop)
            self.long_stop_price = data['Open'].price - data['Open'].price * float(self.target)
            self.order('adj_close', -self.trade_size)
            self.short = True
            self.closed = False
            self.invested = True
            self.trading_day_counter = 0
        if self.invested and (data['adj_close'].price <= self.long_stop_price or data[
            'adj_close'].price >= self.short_stop_price):  # or self.trading_day_counter == HOLDTIME):

            if self.long:
                self.order('adj_close', -self.trade_size)
            if self.short:
                self.order('adj_close', self.trade_size)

            self.closed = True
            self.long = False
            self.short = False
            self.invested = False

        self.trading_day_counter = self.trading_day_counter + 1
        self.record(adj_close=data['adj_close'].price,
                    upper=data['upper'].price,
                    lower=data['lower'].price,
                    long=self.long,
                    short=self.short,
                    holdtime=self.trading_day_counter,
                    closed_position=self.closed,
                    shares=self.trade_size)


if __name__ == '__main__':
    df = fc.get_time_series(ticker='AAPL')  # contains Date, Open, High, Low, adj_close, Volume

    df['avg'] = df['adj_close']['adj_close'].rolling(window=21, center=False).mean()
    df['std'] = df['adj_close'].rolling(window=21, center=False).std()
    df['upper'] = df['avg'] + 2 * df['std']
    df['lower'] = df['avg'] - 2 * df['std']
    df = df.dropna()

    # # # # init Strat Class
    Strategy = BollingerBands()
    # #print df

    # # # # # # Run Strategy
    results = Strategy.run(df)
    results['algorithm_returns'] = (1 + results.returns).cumprod()

    results.to_csv('output.csv')
    print(results['algorithm_returns'].tail(1)[0] * 100)
