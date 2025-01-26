import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")

sales_df = pd.read_csv("sales_data_sample.csv", encoding='latin-1')

print(sales_df.columns)


sales_df.drop_duplicates(inplace=True)

for col in sales_df.select_dtypes(include=np.number):
    sales_df[col] = sales_df[col].fillna(sales_df[col].median())

if 'ORDERDATE' in sales_df.columns:
    sales_df['ORDERDATE'] = pd.to_datetime(sales_df['ORDERDATE'])
    sales_df.rename(columns={'ORDERDATE': 'Date'}, inplace=True)
else:
    print("Error: 'ORDERDATE' column not found in the dataset.")

if 'SALES' not in sales_df.columns:
    print("Error: 'SALES' column not found in the dataset. Time series analysis skipped.")
else:
    
    
    sales_df.groupby('Date')['SALES'].sum().plot(figsize=(12, 6), title="Sales Over Time")
    plt.ylabel("Sales")
    plt.show()

if 'MSRP' in sales_df.columns and 'PROFIT' in sales_df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=sales_df, x="MSRP", y="PROFIT", alpha=0.5)
    plt.title("Profit vs MSRP")
    plt.show()
else:
    print("Error: Columns 'MSRP' and/or 'PROFIT' not found in the dataset.")

if 'Region' in sales_df.columns and 'Category' in sales_df.columns and 'SALES' in sales_df.columns:
    plt.figure(figsize=(10, 8))
    sns.barplot(data=sales_df, x="Region", y="SALES", hue="Category", ci=None)
    plt.title("Sales by Region and Category")
    plt.show()
else:
    print("Error: Columns 'Region', 'Category', and/or 'SALES' not found in the dataset.")


if all(col in sales_df.columns for col in ["PROFIT", "Discount", "SALES"]):
    # Prepare data for Linear Regression
    X = sales_df[["PROFIT", "Discount"]]
    y = sales_df["SALES"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Model Performance:\nR² Score: {r2:.2f}\nMean Squared Error: {mse:.2f}")
else:
    print("Error: Columns 'PROFIT', 'Discount', and/or 'SALES' not found in the dataset. Predictive modeling skipped.")