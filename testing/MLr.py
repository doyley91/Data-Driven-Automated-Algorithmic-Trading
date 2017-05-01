from mlearning import SGDr, SVMr, LinearRegression, BTr, DTr, KNNr, MLPr, RFr
import pandas as pd
from matplotlib import pyplot as plt

ticker = 'AAPL'

SGDr_results = SGDr.run(ticker=ticker)
SVMr_results = SVMr.run(ticker=ticker)
LinearRegression_results = LinearRegression.run(ticker=ticker)
BTr_results = BTr.run(ticker=ticker)
DTr_results = DTr.run(ticker=ticker)
KNNr_results = KNNr.run(ticker=ticker)
MLPr_results = MLPr.run(ticker=ticker)
RFr_results = RFr.run(ticker=ticker)

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
