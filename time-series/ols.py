import functions as fc
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

AAPL = fc.get_time_series('AAPL')

fc.plot_end_of_day(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

AAPL = fc.get_sma_regression_features(AAPL).dropna()

train_size = int(len(AAPL) * 0.80)

train, test = AAPL[0:train_size], AAPL[train_size:len(AAPL)]

mdl = sm.OLS(train['adj_close'], train['sma_15']).fit()
mdl.summary()

mdl.params

mdl.bse

# in sample prediction
pred = mdl.predict(test['sma_15'])

results = pd.DataFrame(data=dict(original=test['adj_close'], prediction=pred), index=test.index)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(results['original'])
ax.plot(results['prediction'])
ax.set(title='In-Sample Return Prediction\nOLS', xlabel='time', ylabel='$')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()
