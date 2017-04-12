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

mdl = sm.OLS(train['adj_close'].values, train['sma_15'].values).fit()
mdl.summary()

mdl.params

mdl.bse

# in sample prediction
pred = mdl.predict(test['sma_15'].values)

# summarize the fit of the model
explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score = fc.get_regression_metrics(test['adj_close'], pred)

results = pd.DataFrame(data=dict(original=test['adj_close'], prediction=pred), index=test.index)

# Plot 21 day forecast for AAPL returns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(results['original'])
ax.plot(results['prediction'])
ax.set(title='OLS In-Sample Return Prediction', xlabel='time', ylabel='$')
ax.legend(['Original', 'Prediction'])
fig.tight_layout()

# out-of-sample test
n_steps = 21
forecast = fc.forecast_regression(model=mdl, sample=test, features='sma_15', steps=n_steps)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(forecast['adj_close'][-n_steps:])
ax.set(title='{} Day Out-of-Sample Forecast'.format(n_steps), xlabel='time', ylabel='$')
ax.legend(['Forecast $'])
fig.tight_layout()
