import functions as fc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

AAPL = fc.get_time_series('AAPL')

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

# fit a Naive Bayes model to the data
mdl = RandomForestClassifier().fit(X, Y)
print(mdl)

# make predictions
pred = mdl.predict(test[features].values)

results = pd.DataFrame(data=dict(original=test['outcome'], prediction=pred), index=test.index)

# summarize the fit of the model
classification_report, confusion_matrix = fc.get_classifier_metrics(results['original'], results['prediction'])

# out-of-sample test
n_steps = 21

forecast = fc.forecast_classifier(model=mdl, sample=test, features=features, steps=n_steps)

