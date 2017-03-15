import functions as fc
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn import metrics
import pydotplus as pydot
import matplotlib.pyplot as plt

AAPL = fc.return_ticker('AAPL')

fc.end_of_day_plot(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# add the outcome variable, 1 if the trading session was positive (close>open), 0 otherwise
AAPL['outcome'] = AAPL.apply(lambda x: 1 if x['adj_close'] > x['adj_open'] else -1, axis=1)

# distance between Highest and Opening price
AAPL['ho'] = AAPL['adj_high'] - AAPL['adj_open']

# distance between Lowest and Opening price
AAPL['lo'] = AAPL['adj_low'] - AAPL['adj_open']

# difference between Closing price - Opening price
AAPL['gain'] = AAPL['adj_close'] - AAPL['adj_open']

AAPL = fc.get_sma_features(AAPL).dropna()

training_set = AAPL[:-500]
test_set = AAPL[-500:]

# values of features
X = np.array(training_set[['sma_15', 'sma_50']].values)

# target values
Y = np.array(training_set['adj_close'])

mdl = DecisionTreeRegressor().fit(X, Y)
print(mdl)

dot_data = export_graphviz(mdl,
                           out_file=None,
                           feature_names=list(training_set[['feat1', 'feat2', 'feat3', 'feat4', 'feat5']]),
                           class_names='outcome',
                           filled=True,
                           rounded=True,
                           special_characters=True)

graph = pydot.graph_from_dot_data(dot_data)
graph.write_png("charts/decision-tree-regression.png")

pred = mdl.predict(test_set[['sma_15', 'sma_50']].values)

metrics.mean_absolute_error(test_set['adj_close'], pred)
metrics.mean_squared_error(test_set['adj_close'], pred)
metrics.median_absolute_error(test_set['adj_close'], pred)
metrics.r2_score(test_set['adj_close'], pred)

results = pd.DataFrame(data=dict(original=test_set['adj_close'], prediction=pred), index=test_set.index)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(results['original'])
ax.plot(results['prediction'])
ax.set(title='Time Series Plot', xlabel='time', ylabel='$')
ax.legend(['Original $', 'Forecast $'])
fig.tight_layout()

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:,1], Y, c="darkorange", label="data")
ax.plot(test_set['adj_close'].values, pred, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
fig.tight_layout()