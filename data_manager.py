#주가데이터를 받아오고 전처리하는 함수
#yfinance or pandas_datareader를 이용해서 주가데이터를 받아옴
#아니면 이미 있는 csv 나 json 파일에서 받아옴
#입력 인자로는 stock code, start date, end date가 있음

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

def preprocessing():
    name = 'AAPL'
    df = pd.read_csv(f'csvfolder/{name}.csv', parse_dates=['Date'], index_col='Date')
    df = df.ffill()  # 경고 수정

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

    def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)

    sequence_length = 60  # 60일 치 데이터를 사용해 예측
    X = create_sequences(scaled_data, sequence_length)
    y = scaled_data[sequence_length:, 3]  # 종가(Close)를 예측

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 데이터 형식 확인
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test

