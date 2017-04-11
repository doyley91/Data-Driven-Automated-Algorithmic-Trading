import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

nsample = 50
sig = 0.25
x1 = np.linspace(0, 20, nsample)
X = np.c_[x1, np.sin(x1), (x1 - 5)**2, np.ones(nsample)]
beta = [0.5, 0.5, -0.02, 5.]
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

mdl = sm.OLS(y, X).fit()
mdl.params
mdl.bse

# predicts y
pred = mdl.predict(X)

x1n = np.linspace(20.5, 25, 10)
Xnew = np.c_[x1n, np.sin(x1n), (x1n - 5)**2, np.ones(10)]
ynewpred = mdl.predict(Xnew)  # predict out of sample

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x1, y, 'o', x1, y_true, 'b-');
ax.plot(np.hstack((x1, x1n)), np.hstack((ypred, ynewpred)), 'r');
ax.title('OLS prediction, blue: true and data, fitted/predicted values:red');
fig.tight_layout()
