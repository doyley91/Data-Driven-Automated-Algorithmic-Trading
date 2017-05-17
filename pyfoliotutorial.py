import functions as fc
import pyfolio as pf

df = fc.get_time_series(ticker='AAPL')

df['returns'] = df['adj_close'].pct_change(periods=1)

stock_rets = df['returns']

pf.create_returns_tear_sheet(returns=stock_rets)
