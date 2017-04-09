import functions as fc
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus as pydot

AAPL = fc.get_time_series('AAPL').round(2)

fc.plot_end_of_day(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# add the outcome variable, 1 if the trading session was positive (close>open), 0 otherwise
AAPL['outcome'] = AAPL.apply(lambda x: 1 if x['adj_close'] > x['adj_open'] else -1, axis=1)

AAPL = fc.get_sma_classifier_features(AAPL)

train_size = int(len(AAPL) * 0.80)

train, test = AAPL[0:train_size], AAPL[train_size:len(AAPL)]

features = ['sma_2', 'sma_3', 'sma_4', 'sma_5', 'sma_6']

# values of features
X = list(train[features].values)

# target values
Y = list(train['outcome'])

mdl = DecisionTreeClassifier().fit(X, Y)
print(mdl)

dot_data = export_graphviz(mdl,
                           out_file=None,
                           feature_names=list(train[['feat1', 'feat2', 'feat3', 'feat4', 'feat5']]),
                           class_names='outcome',
                           filled=True,
                           rounded=True,
                           special_characters=True)

graph = pydot.graph_from_dot_data(dot_data)
graph.write_png("charts/decision-tree-classifier2.png")

pred = mdl.predict(test[features].values)
pred_prob = mdl.predict_proba(test[features].values)

print(metrics.classification_report(test['outcome'], pred))
print(metrics.confusion_matrix(test['outcome'], pred))

results = pd.DataFrame(data=dict(original=test['outcome'], prediction=pred), index=test.index)

# out-of-sample test
n_steps = 21

forecast = fc.forecast_classifier(model=mdl, sample=test, features=features, steps=n_steps)
