import functions as fc
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = fc.get_time_series('AAPL')

fc.plot_end_of_day(df['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# add the outcome variable, 1 if the trading session was positive (close>open), 0 otherwise
df['outcome'] = df.apply(lambda x: 1 if x['adj_close'] > x['adj_open'] else -1, axis=1)

df = fc.get_sma_regression_features(df).dropna()

train_size = int(len(df) * 0.80)

train, test = df[0:train_size], df[train_size:len(df)]

features = ['sma_15', 'sma_50']

# values of features
X = list(train[features].values)

# target values
Y = list(train['outcome'].values)

# fit a Naive Bayes model to the data
mdl = LogisticRegression().fit(X, Y)
print(mdl)

# make predictions
pred = mdl.predict(test[features].values)

# summarize the fit of the model
results = pd.DataFrame(data=dict(original=test['outcome'], prediction=pred), index=test.index)

# summarize the fit of the model
classification_report, confusion_matrix = fc.get_classifier_metrics(results['original'], results['prediction'])

# out-of-sample test
n_steps = 21

forecast = fc.forecast_classifier(model=mdl, sample=test, features=features, steps=n_steps)
