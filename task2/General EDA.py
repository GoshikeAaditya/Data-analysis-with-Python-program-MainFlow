import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn. linear_model import LinearRegression 
from sklearn.metrics import r2_score, mean_squared_error 
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")

df = pd. read_csv ("Global_Superstore2.csv", encoding='latin-1')

numeric_columns = df. select_dtypes (include=np. number). columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

df. drop_duplicates (inplace=True)

def detect_outliers(data, column):
    Q1 = data[column].quantile (0.25)
    Q3 = data[column].quantile (0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

for col in numeric_columns:
    df = detect_outliers(df, col)
    
stats = df. describe ()
numeric_df = df. select_dtypes (include=np. number)
correlation = numeric_df.corr()

df.hist(figsize=(10, 8))
plt. show()

plt. figure(figsize=(10, 6))
sns. boxplot(data=df [numeric_columns], orient="h")
plt. show()

plt. figure(figsize=(10, 8))
sns. heatmap (correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt. show()