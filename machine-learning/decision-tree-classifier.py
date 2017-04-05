import functions as fc
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus as pydot

AAPL = fc.get_time_series('AAPL')

fc.end_of_day_plot(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# add the outcome variable, 1 if the trading session was positive (close>open), 0 otherwise
AAPL['outcome'] = AAPL.apply(lambda x: 1 if x['adj_close'] > x['adj_open'] else -1, axis=1)

# distance between Highest and Opening price
AAPL['ho'] = AAPL['adj_high'] - AAPL['adj_open']

# distance between Lowest and Opening price
AAPL['lo'] = AAPL['adj_low'] - AAPL['adj_open']

# difference between Closing price - Opening price
AAPL['gain'] = AAPL['adj_close'] - AAPL['adj_open']

AAPL = fc.get_sma_classifier_features(AAPL)

training_set = AAPL[:-500]
test_set = AAPL[-500:]

features = ['sma_2', 'sma_3', 'sma_4', 'sma_5', 'sma_6']

# values of features
X = list(training_set[features].values)

# target values
Y = list(training_set['outcome'])

mdl = DecisionTreeClassifier().fit(X, Y)
print(mdl)

dot_data = export_graphviz(mdl,
                           out_file=None,
                           feature_names=list(training_set[['feat1', 'feat2', 'feat3', 'feat4', 'feat5']]),
                           class_names='outcome',
                           filled=True,
                           rounded=True,
                           special_characters=True)

graph = pydot.graph_from_dot_data(dot_data)
graph.write_png("charts/decision-tree-classifier2.png")

pred = mdl.predict(test_set[features].values)
pred_prob = mdl.predict_proba(test_set[features].values)

print(metrics.classification_report(test_set['outcome'], pred))
print(metrics.confusion_matrix(test_set['outcome'], pred))

results = pd.DataFrame(data=dict(original=test_set['outcome'], prediction=pred), index=test_set.index)