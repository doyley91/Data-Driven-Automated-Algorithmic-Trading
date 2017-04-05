import functions as fc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = fc.get_time_series()

df_corr = fc.get_correlated_dataframe(df)

#saving the DataFrame of correlated values to csv
df.to_csv('data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db_corr.csv')

#creating an array of the values of correlations in the DataFrame
data1 = df_corr.values