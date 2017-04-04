import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from matplotlib import mlab
from statsmodels.tsa.stattools import adfuller
from scipy.stats.mstats import normaltest
from scipy.stats import shapiro, kstest, anderson
import talib as ta
from talib import MA_Type
from sklearn.svm import SVC, LinearSVC
import scipy.stats as stats
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
from sqlalchemy import create_engine
import quandl as qdl
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# location of the data set
file_location = "data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"


def return_ticker(ticker):
    """
    returns end-of-day data of the selected ticker in the format of a dataframe 
    if condition checks if one ticker or a list of tickers are being passed
    """

    df = pd.read_csv(file_location, index_col='date', parse_dates=True)
    if isinstance(ticker, list):
        df = df.loc[df['ticker'].isin(ticker)]
        df = df[np.isfinite(df['adj_close'])]
    else:
        df = df.loc[df['ticker'] == ticker]
        df = df[np.isfinite(df['adj_close'])]
    return df


def end_of_day_plot(TS, title=None, xlabel=None, ylabel=None, legend=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(TS)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend([legend])
    fig.tight_layout()


def plot_ticker(TS):
    fig = plt.figure()
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax1.plot(TS['adj_close'])
    ax1.plot(TS['adj_close'].rolling(window=100, min_periods=0).mean())
    ax1.set(title='Time Series Plot', xlabel='time', ylabel='$')
    ax1.legend(['Adjusted Close $', '100 day moving average'])
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    ax2.bar(TS.index, TS['adj_volume'])
    ax2.set(title='Time Series Plot', xlabel='time', ylabel='$')
    ax2.legend(['Volume'])
    fig.tight_layout()


def test_stationarity(TS):
    """
    returns a list of stationary statistics for the dataframe being passed
    """

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


def plot_time_series(y, lags=None):
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


def plot_svm(X, Y, ylabel, xlabel):
    h = .02  # step size in the mesh

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    svc = SVC(kernel='linear', C=C).fit(X, Y)
    rbf_svc = SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
    poly_svc = SVC(kernel='poly', degree=3, C=C).fit(X, Y)
    lin_svc = LinearSVC(C=C).fit(X, Y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']

    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])


def plot_svm_2(X, Y):
    # figure number
    fignum = 1

    # fit the model
    for name, penalty in (('unreg', 1), ('reg', 0.05)):
        clf = SVC(kernel='linear', C=penalty)
        clf.fit(X, Y)

        # get the separating hyperplane
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-5, 5)
        yy = a * xx - (clf.intercept_[0]) / w[1]

        # plot the parallels to the separating hyperplane that pass through the
        # support vectors
        margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
        yy_down = yy + a * margin
        yy_up = yy - a * margin

        # plot the line, the points, and the nearest vectors to the plane
        plt.figure(fignum, figsize=(4, 3))
        plt.clf()
        plt.plot(xx, yy, 'k-')
        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')

        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                    facecolors='none', zorder=10)
        plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

        plt.axis('tight')
        x_min = -4.8
        x_max = 4.2
        y_min = -6
        y_max = 6

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.figure(fignum, figsize=(4, 3))
        plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.xticks(())
        plt.yticks(())
        fignum = fignum + 1


def get_best_ma_model(TS):
    """
    loops through all ma models and returns the best one based on the dataframe passed
    """

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
    """
    loops through all arma models and returns the best one based on the dataframe passed
    """

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
    """
    loops through all arima models and returns the best one based on the dataframe passed
    """

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
    """
    loops through all garch models and returns the best one based on the dataframe passed
    """

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


def get_best_sarimax_model(TS):
    """
    loops through all sarimax models and returns the best one based on the dataframe passed
    """

    best_aic = np.inf
    best_order = None
    best_mdl = None

    pq_rng = range(5)  # [0,1,2,3,4]
    d_rng = range(2)  # [0,1]
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
                except:
                    continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl


def get_technical_analysis_features(TS):
    # calculate a simple moving average of the close prices
    TS['sma_5'] = ta.SMA(np.array(TS['adj_close']), 5)

    # 50 day simple moving average
    TS['sma_50'] = ta.SMA(np.array(TS['adj_close']), 20)

    # calculating bollinger bands, with triple exponential moving average
    TS['upper'], TS['middle'], TS['lower'] = ta.BBANDS(np.array(TS['adj_close']), matype=MA_Type.T3)

    # calculating momentum of the close prices, with a time period of 5
    TS['mom_adj_close'] = ta.MOM(np.array(TS['adj_close']), timeperiod=5)

    # AD - Chaikin A/D Line
    TS['AD'] = ta.AD(np.array(TS['adj_high']),
                     np.array(TS['adj_low']),
                     np.array(TS['adj_close']),
                     np.array(TS['adj_volume']))

    # ADOSC - Chaikin A/D Oscillator
    TS['ADOSC'] = ta.ADOSC(np.array(TS['adj_high']),
                           np.array(TS['adj_low']),
                           np.array(TS['adj_close']),
                           np.array(TS['adj_volume']), fastperiod=3, slowperiod=10)

    # OBV - On Balance Volume
    TS['OBV'] = ta.OBV(np.array(TS['adj_close']), np.array(TS['adj_volume']))

    TS['TRANGE'] = ta.TRANGE(np.array(TS['adj_high']),
                             np.array(TS['adj_low']),
                             np.array(TS['adj_close']))

    return TS


def get_lagged_features(TS):
    """
    generates a lagged time series and returns the features for the classifier
    """

    # generate lagged time series
    TS_1 = TS.shift(1)
    TS_2 = TS.shift(2)
    TS_3 = TS.shift(3)
    TS_4 = TS.shift(4)
    TS_5 = TS.shift(5)

    TS['feat1'] = TS['adj_close'] > TS_1['adj_close']
    TS['feat2'] = TS['adj_close'] > TS_2['adj_close']
    TS['feat3'] = TS['adj_close'] > TS_3['adj_close']
    TS['feat4'] = TS['adj_close'] > TS_4['adj_close']
    TS['feat5'] = TS['adj_close'] > TS_5['adj_close']

    return TS


def get_sma_regression_features(TS):
    """
    calculates the 15 and 50 day simple moving average and returns the features for the regression
    """

    # calculate a simple moving average of the close prices
    TS['sma_15'] = ta.SMA(np.array(TS['adj_close']), 15)

    # 50 day simple moving average
    TS['sma_50'] = ta.SMA(np.array(TS['adj_close']), 50)

    return TS


def get_sma_classifier_features(TS):
    """
    calculates the 2-6 day simple moving average and returns the features for the classifier
    """

    TS['sma_2'] = ta.SMA(np.array(TS['adj_close']), 2)
    TS['sma_3'] = ta.SMA(np.array(TS['adj_close']), 3)
    TS['sma_4'] = ta.SMA(np.array(TS['adj_close']), 4)
    TS['sma_5'] = ta.SMA(np.array(TS['adj_close']), 5)
    TS['sma_6'] = ta.SMA(np.array(TS['adj_close']), 5)

    TS['sma_2'] = TS.apply(lambda x: 1 if x['adj_close'] > x['sma_2'] else 0, axis=1)
    TS['sma_3'] = TS.apply(lambda x: 1 if x['adj_close'] > x['sma_3'] else 0, axis=1)
    TS['sma_4'] = TS.apply(lambda x: 1 if x['adj_close'] > x['sma_4'] else 0, axis=1)
    TS['sma_5'] = TS.apply(lambda x: 1 if x['adj_close'] > x['sma_5'] else 0, axis=1)
    TS['sma_6'] = TS.apply(lambda x: 1 if x['adj_close'] > x['sma_6'] else 0, axis=1)

    return TS


def generate_proj_returns(burn_in, trace, len_to_train):
    num_pred = 1000
    mod_returns = np.ones(shape=(num_pred, len_to_train))
    vol = np.ones(shape=(num_pred, len_to_train))
    for k in range(0, num_pred):
        nu = trace[burn_in + k]['nu']
        mu = trace[burn_in + k]['mu']
        sigma = trace[burn_in + k]['sigma']
        s = trace[burn_in + k]['logs'][-1]
        for j in range(0, len_to_train):
            cur_log_return, s = _generate_proj_returns(mu,
                                                       s,
                                                       nu,
                                                       sigma)
            mod_returns[k, j] = cur_log_return
            vol[k, j] = s
    return mod_returns, vol


def _generate_proj_returns(mu, volatility, nu, sig):
    next_vol = np.random.normal(volatility, scale=sig)  # sig is SD

    # Not 1/np.exp(-2*next_vol), scale treated differently in scipy than pymc3
    log_return = stats.t.rvs(nu, mu, scale=np.exp(-1 * next_vol))
    return log_return, next_vol


def convert_prices_to_log(prices, TS, test_set):
    """
    converts the end of day close prices to log and returns them
    """

    for k in range(0, len(prices)):
        cur = np.log(TS.values[test_set[0]])
        for j in range(0, len(prices[k])):
            cur = cur + prices[k, j]
            prices[k, j] = cur
    return prices


def get_correlated_dataframe(TS):
    """
    pivots the dataframe to return the correlations of the stocks in the dataframe
    """

    # pivoting the DataFrame to create a column for every ticker
    TS = TS.pivot(index=None, columns='ticker', values='adj_close')

    # creating a DataFrame with the correlation values of every column to every column
    TS = TS.corr()

    return TS


def plot_correlation(df_corr):
    """
    plots a heatmap of the correlations between stocks
    """

    # creating an array of the values of correlations in the DataFrame
    data1 = df_corr.values

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    # changes the fontsize of the tick label
    ax1.tick_params(axis='both', labelsize=12)

    # creating a heatmap with colours going from red (negative correlations) to yellow (no correlations) to green (positive correlations)
    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)

    # creating a colour side bar as a scale for the heatmap
    fig1.colorbar(heatmap1)

    # setting the ticks of the x-axis
    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)

    # setting the ticks of the y-axis
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)

    # inverts the scale of the y-axis
    ax1.invert_yaxis()

    # places the x-axis at the top of the graph
    ax1.xaxis.tick_top()

    # storing the ticker labels in an array
    column_labels = df_corr.columns

    # storing the dates in an array
    row_labels = df_corr.index

    # setting the x-axis labels to the dates
    ax1.set_xticklabels(column_labels)

    # setting the y-axis labels to the ticker labels
    ax1.set_yticklabels(row_labels)

    # rotates the x-axis labels vertically to fit the graph
    plt.xticks(rotation=90)

    # sets the range from -1 to 1
    heatmap1.set_clim(-1, 1)

    # automatically adjusts subplot paramaters to give specified padding
    plt.tight_layout()


def plot_candlestick(TS):
    # creating a new DataFrame based on the adjusted_close price resampled with a 10 day window
    tickers_ohlc = TS['adj_close'].resample('10D').ohlc()

    # creating a new DataFrame based on the adjusted volume resampled with a 10 day window
    tickers_volume = TS['adj_volume'].resample('10D').sum()

    # resetting the index of the DataFrame
    tickers_ohlc = tickers_ohlc.reset_index()

    # converting the date column to mdates
    tickers_ohlc['date'] = tickers_ohlc['date'].map(mdates.date2num)

    # creating a new figure
    fig = plt.figure()

    # creating a subplot with a 6x1 grid and starts at (0,0)
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)

    # creating a subplot with a 6x1 grid and starts at (5,0)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)

    # converts the axis from raw mdate numbers to dates
    ax1.xaxis_date()

    # plotting the candlestick graph
    candlestick_ohlc(ax1, tickers_ohlc.values, width=2, colorup='g')

    # plotting the volume bar chart
    ax2.fill_between(tickers_volume.index.map(mdates.date2num), tickers_volume.values, 0)

    fig.tight_layout()


def forecast_classifier(model, sample, features, steps=1):
    for k in range(1, steps):
        sample.index = sample.index = pd.DateOffset(1)
        sample['outcome'][-1:] = model.predict(sample[features[-2:][:1]])
        sample = get_sma_classifier_features(sample)
    return sample


def forecast_regression(model, sample, features, steps=1):
    for k in range(1, steps):
        sample.index = sample.index + pd.DateOffset(1)
        sample['adj_close'][-1:] = model.predict(sample[features][-2:][:1])
        sample = get_sma_regression_features(sample)
    return sample


def export_to_sql():
    DB_TYPE = 'postgresql'
    DB_DRIVER = 'psycopg2'
    DB_USER = 'admin'
    DB_PASS = 'password'
    DB_HOST = 'localhost'
    DB_PORT = '5432'
    DB_NAME = 'pandas_upsert'
    POOL_SIZE = 50
    TABLENAME = 'test_upsert'
    SQLALCHEMY_DATABASE_URI = '%s+%s://%s:%s@%s:%s/%s' % (DB_TYPE, DB_DRIVER, DB_USER,
                                                          DB_PASS, DB_HOST, DB_PORT, DB_NAME)

    ENGINE = create_engine(SQLALCHEMY_DATABASE_URI, pool_size=POOL_SIZE, max_overflow=0)

    pd.to_sql(TABLENAME, ENGINE, if_exists='append', index=False)


def download_data():
    # quandl.get(":database_code/:dataset_code", returns = ":return_format")
    qdl.ApiConfig.api_key = 'GY56ffY8tRJKuZyfYVsH'

    # When returns is omitted, a pandas dataframe is returned
    data = qdl.get("WIKI/FB")

    # quandl.Dataset(":database_code/:dataset_code").data_fields()
    metadata = qdl.Dataset("WIKI/FB").data_fields()

    # Multiple datasets in one call, comma-delimit their codes and put them in an array
    merged_data = qdl.get(["WIKI/FB", "EOD/AAPL", "WIKI/MSFT"])