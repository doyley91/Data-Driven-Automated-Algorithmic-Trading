import pandas as pd
from matplotlib import pyplot as plt

from mlearning import SGDr, SVMr, LinearRegression, BTr, DTr, KNNr, MLPr, RFr

tickers = 'AAPL'

SGDr_results = SGDr.main(tickers=tickers)
SVMr_results = SVMr.main(tickers=tickers)
LinearRegression_results = LinearRegression.main(tickers=tickers)
BTr_results = BTr.main(tickers=tickers)
DTr_results = DTr.main(tickers=tickers)
KNNr_results = KNNr.main(tickers=tickers)
MLPr_results = MLPr.main(tickers=tickers)
RFr_results = RFr.main(tickers=tickers)

regression_results = pd.DataFrame(data=dict(SGD=SGDr_results.values,
                                            SVM=SVMr_results.values,
                                            LinearRegression=LinearRegression_results.values,
                                            BT=BTr_results.values,
                                            DT=DTr_results.values,
                                            KNN=KNNr_results.values,
                                            MLP=MLPr_results.values,
                                            RF=RFr_results.values),
                                  index=SGDr_results.index)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(regression_results['SGD'])
ax.plot(regression_results['SVM'])
ax.plot(regression_results['LinearRegression'])
ax.plot(regression_results['BT'])
ax.plot(regression_results['DT'])
ax.plot(regression_results['KNN'])
ax.plot(regression_results['MLP'])
ax.plot(regression_results['RF'])
ax.set(title='ML Out-Of-Sample Forecast', xlabel='time', ylabel='$')
ax.legend(['SGD', 'SVM', 'Linear Regression', 'BT', 'DT', 'KNN', 'MLP', 'RF'])
fig.tight_layout()
