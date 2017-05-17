import pandas as pd

from mlearning import SGDc, SVMc, Logit, BTc, DTc, KNNc, MLPc, RFc, NBBc, NBGc

ticker = 'AAPL'

SGDc_results = SGDc.run(ticker=ticker)
SVMc_results = SVMc.run(ticker=ticker)
Logit_results = Logit.run(ticker=ticker)
BTc_results = BTc.run(ticker=ticker)
DTc_results = DTc.run(ticker=ticker)
KNNc_results = KNNc.run(ticker=ticker)
MLPc_results = MLPc.run(ticker=ticker)
RFc_results = RFc.run(ticker=ticker)
NBBc_results = NBBc.run(ticker=ticker)
NBGc_results = NBGc.run(ticker=ticker)

regression_results = pd.DataFrame(data=dict(SGD=SGDc_results.values,
                                            SVM=SVMc_results.values,
                                            Logit=Logit_results.values,
                                            BT=BTc_results.values,
                                            DT=DTc_results.values,
                                            KNN=KNNc_results.values,
                                            MLP=MLPc_results.values,
                                            RF=RFc_results.values),
                                  index=SGDc_results.index)
