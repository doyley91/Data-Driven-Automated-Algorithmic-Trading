import statsmodels.api as sm
import pandas

data = sm.datasets.sunspots.load()

dates = sm.tsa.datetools.dates_from_range('1700', length=len(data.endog))

endog = pandas.TimeSeries(data.endog, index=dates)

ar_model = sm.tsa.AR(endog, freq='A')
pandas_ar_res = ar_model.fit(maxlag=9, method='mle', disp=-1)

pred = pandas_ar_res.predict(start='2005', end='2015')

ar_model = sm.tsa.AR(data.endog, dates=dates, freq='A')
ar_res = ar_model.fit(maxlag=9, method='mle', disp=-1)
pred = ar_res.predict(start='2005', end='2015')
