# data_manager.py

import pandas as pd

def load_data(file_path, start_date=None, end_date=None):
    """
    Load stock data from a CSV file within a specified date range.

    Args:
        file_path (str): The path to the CSV file containing the stock data.
        start_date (str): The start date for filtering the data (format: 'YYYY-MM-DD').
        end_date (str): The end date for filtering the data (format: 'YYYY-MM-DD').

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Filter data by date range if specified
    if start_date:
        data = data[data.index >= start_date]
    if end_date:
        data = data[data.index <= end_date]
    
    chart_data, train_data = preprocess_data(data)
    return chart_data, train_data

def preprocess_data(data):
    """
    Preprocess the stock data for reinforcement learning.

    Args:
        data (pd.DataFrame): The raw stock data.

    Returns:
        pd.DataFrame, pd.DataFrame: The processed chart data and training data.
    """
    # Ensure all necessary columns are present
    # required_columns = [
    #     'Open', 'High', 'Low', 'Close', 'Volume',
    #     'Open_SMA_5', 'Open_SMA_10', 'Open_SMA_20', 'Open_SMA_60', 'Open_SMA_120',
    #     'Close_SMA_5', 'Close_SMA_10', 'Close_SMA_20', 'Close_SMA_60', 'Close_SMA_120',
    #     'High_SMA_5', 'High_SMA_10', 'High_SMA_20', 'High_SMA_60', 'High_SMA_120',
    #     'Low_SMA_5', 'Low_SMA_10', 'Low_SMA_20', 'Low_SMA_60', 'Low_SMA_120',
    #     'Volume_SMA_5', 'Volume_SMA_10', 'Volume_SMA_20', 'Volume_SMA_60', 'Volume_SMA_120',
    #     'RSI_14', 'MACD', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'OBV',
    #     'Dividends', 'Splits', 'Interest_Rate', 'Inflation_Rate', 'Consumer_Sentiment'
    # ]
    required_columns = [
        'Open',
        'High',
        'Low',
        'Close',
        'Volume',
        'ratio_to_Close_MA_5',
        'ratio_to_Close_MA_10',
        'ratio_to_Close_MA_20',
        'ratio_to_Close_MA_60',
        'ratio_to_Close_MA_120',
        'RSI_14',
        'MACD',
        'BB_Middle',
        'BB_Upper',
        'BB_Lower',
        'OBV',
        'open_lastclose_ratio',
        'high_close_ratio',
        'low_close_ratio',
        'close_lastclose_ratio',
    ]

    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f'Missing required column: {col}')

    # Fill any remaining NaNs in the dataset
    data.fillna(0, inplace=True)

    # Chart data contains basic stock information for each day
    chart_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Training data contains the additional features
    training_data = data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    return chart_data, training_data
