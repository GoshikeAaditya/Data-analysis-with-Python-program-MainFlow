import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_synthetic_stock_data():
    
    date_range = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
    
    np.random.seed(42)
    n_days = len(date_range)
    base_price = 100  

    close_prices = base_price + np.cumsum(np.random.normal(0, 1, n_days))
    
    open_prices = close_prices + np.random.normal(0, 0.5, n_days)
    high_prices = close_prices + np.abs(np.random.normal(0, 1, n_days))
    low_prices = close_prices - np.abs(np.random.normal(0, 1, n_days))
    
    volumes = np.random.randint(100000, 1000000, n_days)
    
    df = pd.DataFrame({
        'Date': date_range,
        'Open': open_prices,
        'Close': close_prices,
        'High': high_prices,
        'Low': low_prices,
        'Volume': volumes
    })
    
    df.set_index('Date', inplace=True)
    
    df.to_csv('synthetic_stock_prices.csv')
    
    return df

def load_and_preprocess_data():
    df = pd.read_csv('synthetic_stock_prices.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def perform_eda(df):
    df['Close'].plot(figsize=(10, 6))
    plt.title('Stock Close Prices Over Time')
    plt.show()

def feature_engineering(df):
    df['Lag_1'] = df['Close'].shift(1)
    df['Rolling_Mean'] = df['Close'].rolling(window=5).mean()
    df.dropna(inplace=True)
    
    return df

def train_arima_model(df):
    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit

def evaluate_model(model_fit, df):
    forecast = model_fit.forecast(steps=30)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Actual')
    plt.plot(pd.date_range(df.index[-1], periods=30, freq='D'), forecast, label='Forecast')
    plt.title('Stock Price Forecast')
    plt.legend()
    plt.show()
    
    mae = mean_absolute_error(df['Close'][-30:], forecast)
    rmse = np.sqrt(mean_squared_error(df['Close'][-30:], forecast))
    print(f'MAE: {mae}, RMSE: {rmse}')

def main():

    df = create_synthetic_stock_data()
    

    df = load_and_preprocess_data()
    

    perform_eda(df)
    

    df = feature_engineering(df)
    

    model_fit = train_arima_model(df)
    

    evaluate_model(model_fit, df)

if __name__ == "__main__":
    main()