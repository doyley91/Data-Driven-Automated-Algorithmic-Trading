import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

#location of the data set
file_location = "data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"

# importing the data set, converting date column to datetime, making the trading date the index for the Pandas DataFrame and sorting the DataFrame by date
df = pd.read_csv(file_location, index_col='date', parse_dates=True)

#pivoting the DataFrame to create a column for every ticker
pdf = df.pivot(index=None, columns='ticker', values='adj_close')

#creating a DataFrame with the correlation values of every column to every column
df_corr = pdf.corr()

#saving the DataFrame of correlated values to csv
df_corr.to_csv('data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db_corr.csv')

#creating an array of the values of correlations in the DataFrame
data1 = df_corr.values

#creating a new figure
fig1 = plt.figure()

#creating an axis
ax1 = fig1.add_subplot(111)

#changes the fontsize of the tick label
ax1.tick_params(axis='both', labelsize=12)

#creating a heatmap with colours going from red (negative correlations) to yellow (no correlations) to green (positive correlations)
heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)

#creating a colour side bar as a scale for the heatmap
fig1.colorbar(heatmap1)

#setting the ticks of the x-axis
ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)

#setting the ticks of the y-axis
ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)

#inverts the scale of the y-axis
ax1.invert_yaxis()

#places the x-axis at the top of the graph
ax1.xaxis.tick_top()

#storing the ticker labels in an array
column_labels = df_corr.columns

#storing the dates in an array
row_labels = df_corr.index

#setting the x-axis labels to the dates
ax1.set_xticklabels(column_labels)

#setting the y-axis labels to the ticker labels
ax1.set_yticklabels(row_labels)

#rotates the x-axis labels vertically to fit the graph
plt.xticks(rotation=90)

#sets the range from -1 to 1
heatmap1.set_clim(-1, 1)

#automatically adjusts subplot paramaters to give specified padding
plt.tight_layout()

#shows the plot
plt.show()

#saves the plot with a dpi of 300
plt.savefig("charts/correlations2.png", dpi=300)