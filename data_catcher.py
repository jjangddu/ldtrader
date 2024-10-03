import yfinance as yf
import pandas as pd
import numpy as np

from fredapi import Fred
import quandl


# API Key 설정
FRED_API_KEY = '41ca203747cb9393676ffb5758f99363'
quandl.ApiConfig.api_key = 'fhQeYFfw5WtD9NxS97Hi'

# 1. Fetch Coca-Cola stock data
ticker = 'KO'
data = yf.download(ticker, start='2020-01-01', end='2023-10-01')

# 2. Calculate Moving Averages for Open, Close, High, Low, Volume
for col in ['Open', 'Close', 'High', 'Low', 'Volume']:
    data[f'{col}_SMA_5'] = data[col].rolling(window=5).mean()
    data[f'{col}_SMA_10'] = data[col].rolling(window=10).mean()
    data[f'{col}_SMA_20'] = data[col].rolling(window=20).mean()
    data[f'{col}_SMA_60'] = data[col].rolling(window=60).mean()
    data[f'{col}_SMA_120'] = data[col].rolling(window=120).mean()

# 3. Calculate RSI (Relative Strength Index)
delta = data['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI_14'] = 100 - (100 / (1 + rs))

# 4. Calculate MACD (Moving Average Convergence Divergence)
ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26

# 5. Calculate Bollinger Bands
data['BB_Middle'] = data['Close'].rolling(window=20).mean()
data['BB_Upper'] = data['BB_Middle'] + (data['Close'].rolling(window=20).std() * 2)
data['BB_Lower'] = data['BB_Middle'] - (data['Close'].rolling(window=20).std() * 2)

# 6. Calculate OBV (On-Balance Volume)
data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

# 7. Add Dividends and Splits
dividends = yf.Ticker(ticker).dividends
splits = yf.Ticker(ticker).splits
data['Dividends'] = dividends.reindex(data.index, fill_value=0)
data['Splits'] = splits.reindex(data.index, fill_value=0)

# 8. Fetch Economic Indicators from FRED (Interest Rate, Inflation Rate)
fred = Fred(api_key=FRED_API_KEY)
data['Interest_Rate'] = fred.get_series('FEDFUNDS', start='2020-01-01', end='2023-10-01').reindex(data.index, method='ffill')
data['Inflation_Rate'] = fred.get_series('CPIAUCSL', start='2020-01-01', end='2023-10-01').pct_change().reindex(data.index, method='ffill')

# 9. Fetch Consumer Sentiment Index from Fred
consumer_sentiment = fred.get_series('UMCSENT', start='2020-01-01', end='2023-10-01')
# Reindex the consumer sentiment data to match the stock data index and forward fill missing values
data['Consumer_Sentiment'] = consumer_sentiment.reindex(data.index, method='ffill')


# 10. Fill NaN values with 0 and save to CSV
data.fillna(0, inplace=True)
data.to_csv('coca_cola_stock_with_moving_averages.csv')

# Display the last few rows of the dataset
print(data.tail())