from mlearning import SGDr, SVMr, LinearRegression, BTr, DTr, KNNr, MLPr, RFr
import pandas as pd

ticker = 'AAPL'

SGDr_results = SGDr.run(ticker)
SVMr_results = SVMr.run(ticker)
LinearRegression_results = LinearRegression.run(ticker)
BTr_results = BTr.run(ticker)
DTr_results = DTr.run(ticker)
KNNr_results = KNNr.run(ticker)
MLPr_results = MLPr.run(ticker)
RFr_results = RFr.run(ticker)

regression_results = pd.DataFrame(data=dict(SGD=SGDr_results,
                                            SVM=SVMr_results,
                                            LinearRegression=LinearRegression_results,
                                            BT=BTr_results,
                                            DT=DTr_results,
                                            KNN=KNNr_results,
                                            MLP=MLPr_results,
                                            RF=RFr_results),
                                  index=SGDr_results.index)
