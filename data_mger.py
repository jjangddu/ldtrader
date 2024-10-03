import pandas as pd
import numpy as np

def preprocess_stock_data(data):
    # 이동 평균 계산에 사용할 윈도우 크기
    windows = [5, 10, 20, 60, 120]

    # 이동 평균 계산 및 비율 계산
    for window in windows:
        data[f'close_ma{window}'] = data['close'].rolling(window).mean()
        data[f'volume_ma{window}'] = data['volume'].rolling(window).mean()
        data[f'close_ma{window}_ratio'] = \
            (data['close'] - data[f'close_ma{window}']) / data[f'close_ma{window}']
        data[f'volume_ma{window}_ratio'] = \
            (data['volume'] - data[f'volume_ma{window}']) / data[f'volume_ma{window}']

    # 추가적인 비율 데이터 계산
    data['open_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastclose_ratio'] = \
        (data['open'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['high_close_ratio'] = (data['high'].values - data['close'].values) / data['close'].values
    data['low_close_ratio'] = (data['low'].values - data['close'].values) / data['close'].values
    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = \
        (data['close'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = (
        (data['volume'][1:].values - data['volume'][:-1].values)
        / data['volume'][:-1].replace(to_replace=0, method='ffill')\
            .replace(to_replace=0, method='bfill').values
    )

    # 차트 데이터 (필요한 열만 선택)
    chart_data = data[['date', 'open', 'high', 'low', 'close', 'volume']]

    # 학습 데이터 (계산된 모든 비율 데이터 포함)
    training_data = data[['open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
                          'close_lastclose_ratio', 'volume_lastvolume_ratio'] +
                         [f'close_ma{window}_ratio' for window in windows] +
                         [f'volume_ma{window}_ratio' for window in windows]]

    return chart_data, training_data