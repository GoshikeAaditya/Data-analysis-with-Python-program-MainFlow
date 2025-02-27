import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/Users/aaditya/Downloads/sales_data_sample.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Plot sales data
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Sales'], label='Sales')
plt.title('Sales Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Fit ARIMA model
model = ARIMA(df['Sales'], order=(5, 1, 0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=30)

# Evaluate
mse = mean_squared_error(test_data, forecast)
rmse = mse**0.5
print(f'RMSE: {rmse}')