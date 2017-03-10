import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from matplotlib import mlab
from statsmodels.tsa.stattools import adfuller
from scipy.stats.mstats import normaltest
from scipy.stats import shapiro, kstest, anderson
import matplotlib.pyplot as plt

#location of the data set
file_location = "data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"

def return_ticker(ticker):
    df = pd.read_csv(file_location, index_col='date', parse_dates=True)
    if ticker is pd.Series:
        df = df.loc[df['ticker'].isin(ticker)]
        df = df[np.isfinite(df['adj_close'])]
    else:
        df = df.loc[df['ticker'] == ticker]
        df = df[np.isfinite(df['adj_close'])]
    return df

def test_stationarity(TS):
    # verify stationarity
    adfstat, pvalue, critvalues, resstore = adfuller(TS, regression="nc", store=True, regresults=True)

    # D’Agostino and Pearson normality test of returns
    dagostino_results = normaltest(TS)

    # Shapiro-Wilk normality test
    shapiro_results = shapiro(TS)

    # Kolmogorov-Smirnov normality test
    ks_results = kstest(TS, cdf='norm')

    # Anderson-Darling normality test
    anderson_results = anderson(TS)

    return adfstat, pvalue, critvalues, resstore, dagostino_results, shapiro_results, ks_results, anderson_results

def histplot(y):
    mu = np.mean(y)  # mean of distribution
    sigma = np.std(y)  # standard deviation of distribution
    x = mu + sigma * np.random.randn(10000)
    # the histogram of the data
    n, bins, patches = plt.hist(x, bins=50, normed=1, facecolor='green', alpha=0.5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel('Returns')
    plt.ylabel('Probability')
    plt.title('Histogram of returns: $\mu={}$, $\sigma={}$'.format(mu, sigma))
    plt.tight_layout()

def tsplot(y, lags=None):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    fig = plt.figure()
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    pp_ax = plt.subplot2grid(layout, (2, 1))

    y.plot(ax=ts_ax)
    ts_ax.set_title('Time Series Analysis Plots')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
    sm.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('QQ Plot')
    scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

    plt.tight_layout()

def get_best_ma_model(TS):
    best_aic = np.inf
    best_order = None
    best_mdl = None

    rng = range(5)  # [0,1,2,3,4,5]
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(TS, order=(0, j)).fit(maxlag=30, method='mle', trend='nc')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (0, j)
                best_mdl = tmp_mdl
        except:
            continue

    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl

def get_best_arma_model(TS):
    best_aic = np.inf
    best_order = None
    best_mdl = None

    rng = range(5)  # [0,1,2,3,4,5]
    for i in rng:
        for j in rng:
            try:
                tmp_mdl = smt.ARMA(TS, order=(i, j)).fit(method='mle', trend='nc')
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (i, j)
                    best_mdl = tmp_mdl
            except:
                continue

    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl

def get_best_arima_model(TS):
    # Fit ARIMA(p, d, q) model to SPY Returns
    # pick best order and final model based on aic

    best_aic = np.inf
    best_order = None
    best_mdl = None

    pq_rng = range(5)  # [0,1,2,3,4]
    d_rng = range(2)  # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(TS, order=(i, d, j)).fit(method='mle', trend='nc')
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except:
                    continue

    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl

def get_best_garch_model(TS):
    best_aic = np.inf
    best_order = None
    best_mdl = None

    pq_rng = range(5) # [0,1,2,3,4]
    d_rng = range(2) # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(TS, order=(i, d, j)).fit(method='mle', trend='nc')
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl

def get_best_sarimax_model(TS):
    best_aic = np.inf
    best_order = None
    best_mdl = None

    pq_rng = range(5) # [0,1,2,3,4]
    d_rng = range(2) # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = sm.tsa.SARIMAX(TS, order=(i, d, j)).fit(mle_regression=True, trend='nc')
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl