import functions as fc
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC

AAPL = fc.get_time_series('AAPL')

fc.end_of_day_plot(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# add the outcome variable, 1 if the trading session was positive (close>open), 0 otherwise
AAPL['outcome'] = AAPL.apply(lambda x: 1 if x['adj_close'] > x['adj_open'] else 0, axis=1)

AAPL = fc.get_sma_classifier_features(AAPL)

training_set = AAPL[:-500]
test_set = AAPL[-500:]

features = ['sma_2', 'sma_3', 'sma_4', 'sma_5', 'sma_6']

# values of features
X = np.array(training_set[features].values)

# target values
Y = np.array(training_set['outcome'])

# fit a SVM model to the data
mdl = SVC().fit(X, Y)
print(mdl)

# make predictions
pred = mdl.predict(test_set[features].values)

# summarize the fit of the model
print(metrics.classification_report(test_set['outcome'], pred))
print(metrics.confusion_matrix(test_set['outcome'], pred))

results = pd.DataFrame(data=dict(original=test_set['outcome'], prediction=pred), index=test_set.index)

# using just 2 features
X = np.array(training_set[features].values)

fc.plot_svm_2(X, Y)

# out-of-sample test
n_steps = 21
forecast = fc.forecast_classifier(model=mdl, sample=test_set, features=features, steps=n_steps)
