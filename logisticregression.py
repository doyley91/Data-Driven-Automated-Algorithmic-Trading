import pandas as pd
from sklearn.linear_model import LogisticRegression

# read the data in
df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")

# rename the 'rank' column because there is also a DataFrame method called 'rank'
df.columns = ["admit", "gre", "gpa", "prestige"]

dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')

cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
data['intercept'] = 1.0

train_size = int(len(data) * 0.80)

train, test = data[0:train_size], data[train_size:len(data)]

train_cols = data.columns[1:]

X = train_cols[train_cols]

Y = train['admit']

mdl = LogisticRegression().fit(X, Y)

print(mdl)

pred = mdl.predict(test[train_cols])
