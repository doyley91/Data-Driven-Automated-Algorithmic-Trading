import functions as fc
from sklearn import metrics
from sklearn.cluster import KMeans

AAPL = fc.return_ticker('AAPL')

fc.end_of_day_plot(AAPL['adj_close'], title='AAPL', xlabel='time', ylabel='$', legend='Adjusted Close $')

# add the outcome variable, 1 if the trading session was positive (close>open), 0 otherwise
AAPL['outcome'] = AAPL.apply(lambda x: 1 if x['adj_close'] > x['adj_open'] else 0, axis=1)

# distance between Highest and Opening price
AAPL['ho'] = AAPL['adj_high'] - AAPL['adj_open']

# distance between Lowest and Opening price
AAPL['lo'] = AAPL['adj_low'] - AAPL['adj_open']

# difference between Closing price - Opening price
AAPL['gain'] = AAPL['adj_close'] - AAPL['adj_open']

# generate lagged time series
AAPL_1 = AAPL.shift(1)
AAPL_2 = AAPL.shift(2)
AAPL_3 = AAPL.shift(3)
AAPL_4 = AAPL.shift(4)
AAPL_5 = AAPL.shift(5)

AAPL['0'] = AAPL['adj_close'] > AAPL_1['adj_close']
AAPL['1'] = AAPL['adj_close'] > AAPL_2['adj_close']
AAPL['2'] = AAPL['adj_close'] > AAPL_3['adj_close']
AAPL['3'] = AAPL['adj_close'] > AAPL_4['adj_close']
AAPL['4'] = AAPL['adj_close'] > AAPL_5['adj_close']

training_set = AAPL[:-500]
test_set = AAPL[-500:]

# values of features
X = list(training_set[['0', '1', '2', '3', '4']].values)

# target values
Y = list(training_set['outcome'])

# fit a SVM model to the data
mdl = KMeans(n_clusters=5, random_state=1).fit(X)
print(mdl)

# make predictions
pred = mdl.predict(test_set[['0', '1', '2', '3', '4']].values)

# summarize the fit of the model
metrics.calinski_harabaz_score(X, mdl.labels_)