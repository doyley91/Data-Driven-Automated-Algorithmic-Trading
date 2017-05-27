import pandas as pd

from mlearning import SGDc, SVMc, Logit, BTc, DTc, KNNc, MLPc, RFc, NBBc, NBGc

ticker = 'AAPL'

SGDc_results = SGDc.main(ticker=ticker)
SVMc_results = SVMc.main(ticker=ticker)
Logit_results = Logit.main(ticker=ticker)
BTc_results = BTc.main(ticker=ticker)
DTc_results = DTc.main(ticker=ticker)
KNNc_results = KNNc.main(ticker=ticker)
MLPc_results = MLPc.main(ticker=ticker)
RFc_results = RFc.main(ticker=ticker)
NBBc_results = NBBc.main(ticker=ticker)
NBGc_results = NBGc.main(ticker=ticker)

regression_results = pd.DataFrame(data=dict(SGD=SGDc_results.values,
                                            SVM=SVMc_results.values,
                                            Logit=Logit_results.values,
                                            BT=BTc_results.values,
                                            DT=DTc_results.values,
                                            KNN=KNNc_results.values,
                                            MLP=MLPc_results.values,
                                            RF=RFc_results.values),
                                  index=SGDc_results.index)
