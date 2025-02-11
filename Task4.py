import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

file_path = "Housing.csv"
df = pd.read_csv(file_path)

numerical_features = ["area", "bedrooms"]
categorical_features = ["mainroad", "guestroom", "basement", "hotwaterheating", 
                        "airconditioning", "prefarea", "furnishingstatus"]
target_variable = "price"

df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

scaler = MinMaxScaler()
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

X = df_encoded.drop(columns=[target_variable])
y = df_encoded[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Root Mean Square Error (RMSE):", rmse)
print("R² Score:", r2)

plt.figure(figsize=(10,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(y_test - y_pred, bins=30, kde=True)
plt.xlabel("Residuals")
plt.title("Residuals Distribution")
plt.show()
