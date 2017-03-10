import pandas as pd
import numpy as np
from collections import Counter
from sklearn import svm, model_selection, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from statistics import mean

#location of the data set
file_location = "data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"

# importing the data set
df = pd.read_csv(file_location)

# converting date column to datetime
df.date = pd.to_datetime(df.date)

# sorting the DataFrame by date
df.sort_values(by='date')

# making the trading date the index for the Pandas DataFrame
df.set_index('date', inplace=True)

def process_data_for_labels(ticker):
    # how many days
    days = 7

    #creating a list of tickers from the DataFrame
    stocks = pdf.columns.values.tolist()

    #filling NaN values in the DataFrame with 0
    pdf.fillna(0, inplace=True)

    #looping over number of days to create new columns for each ticker in the DataFrame
    for i in range(1, days+1):
            pdf['{}_{}d'.format(ticker, i)] = (pdf[ticker].shift(-i) - pdf[ticker]) / pdf[ticker]

    #filling NaN values in the DataFrame with 0
    pdf.fillna(0, inplace=True)

    return stocks, pdf

def buy_sell_hold(*args):
    #columns in the DataFrame
    cols = [c for c in args]

    #fluctuation in future stock price
    requirement = 0.02

    #loop through all the columns
    for col in cols:
        #
        if col > requirement:
            return 1
        if col < requirement:
            return 0
    return 0

def extract_featuresets(ticker):
    #retrieves the list of stocks and the pivoted DataFrame
    stocks, pdf = process_data_for_labels(ticker)

    #creates a new DataFrame with a target column for the prediction algorithm
    pdf['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              pdf['{}_1d'.format(ticker)],
                                              pdf['{}_2d'.format(ticker)],
                                              pdf['{}_3d'.format(ticker)],
                                              pdf['{}_4d'.format(ticker)],
                                              pdf['{}_5d'.format(ticker)],
                                              pdf['{}_6d'.format(ticker)],
                                              pdf['{}_7d'.format(ticker)]))

    #
    vals = pdf['{}_target'.format(ticker)].values.tolist()

    #
    str_vals = [str(i) for i in vals]

    #
    print('Data spread:', Counter(str_vals))

    #filling NaN values in the DataFrame with 0
    pdf.fillna(0, inplace=True)

    #replacing infinite values in the DataFrame with NaN values
    pdf = pdf.replace([np.inf, -np.inf], np.nan)

    #dropping all NaN values
    pdf.dropna(inplace=True)

    #converting stock prices to % changes
    pdf_vals = pdf[[ticker for ticker in tickers]].pct_change()

    #replacing infinite values in the DataFrame with NaN values
    pdf_vals = pdf_vals.replace([np.inf, -np.inf], 0)

    # filling NaN values in the DataFrame with 0
    pdf_vals.fillna(0, inplace=True)

    #daily % changes for every stock on the DataFrame
    X = pdf_vals.values

    #target
    y = pdf['{}_target'.format(ticker)].values

    return X, y, pdf


def do_ml(ticker):
    #retrieving the daily % changes, #target, and the pivoted DataFrame
    X, y, df = extract_featuresets(ticker)

    #shuffles the data so that it is not in any order and creates training and testing samples
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

    #selects a classifier to use
    clf = neighbors.KNeighborsClassifier()

    #training the classifier on the data by fitting X data to y data for each X's and y's
    clf.fit(X_train, y_train)

    #takes the featuresets in X_test and sees if the predictions match the labels in y_test and returns a decimal percentage
    confidence = clf.score(X_test, y_test)

    #prints the accuracy percentage
    print('accuracy:', confidence)

    #uses the predictions of the X_test data and outputs the distribution
    predictions = clf.predict(X_test)

    #primts the accuracy percentage
    print('predicted class counts:', Counter(predictions))

    #applies a voting classifier instead of a K Nearest Neighbors classifier
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    return confidence

def do_ml_all():
    #creates an empty array for our accuracy percentages
    accuracies = []

    #retrieves a list of tickers from the DataFrame
    tickers = df["ticker"].tolist()

    #loops through all the tickers in the list
    for count, ticker in enumerate(tickers):
        #checks if count id divisible by 10
        if count % 10 == 0:
            #prints the count every 10 tickers
            print(count)

        #calculates the accuracy for the current ticker
        accuracy = do_ml(ticker)

        #adds the accuracy percentage to the array
        accuracies.append(accuracy)

        #prints accuracy and average accuracy for every ticker
        print("{} accuracy: {}. Average accuracy:{}".format(ticker, accuracy, mean(accuracies)))