import pandas as pd
import numpy as np
import yfinance as yf
import os
from src.indicators import add_all_indicators

def download_data(ticker: str = 'GC=F', start_date: str = '2008-01-01', end_date: str = None) -> pd.DataFrame:
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/gold_raw_data.csv')
    print("Raw data saved to data/raw/gold_raw_data.csv")
    return df

def create_target_and_lag_features(df: pd.DataFrame, threshold: float = 0.003) -> pd.DataFrame:
    df = df.copy()
    df['Next_Day_Return'] = df['Close'].pct_change().shift(-1)
    conditions = [
        (df['Next_Day_Return'] > threshold),
        (df['Next_Day_Return'] < -threshold)
    ]
    choices = [1, 0]
    df['Target'] = np.select(conditions, choices, default=-1)
    df = df.drop(columns=['Next_Day_Return'])
    df = df.dropna(subset=['Target'])
    return df

def filter_neutral_rows(df: pd.DataFrame) -> pd.DataFrame:
    initial_len = len(df)
    filtered_df = df[df['Target'] != -1].copy()
    final_len = len(filtered_df)
    print(f"Dropped {initial_len - final_len} NEUTRAL rows.")
    return filtered_df

def prepare_pipeline(ticker='GC=F'):
    df_raw = download_data(ticker)
    df_indicators = add_all_indicators(df_raw)
    df_indicators = df_indicators.dropna()
    df_target = create_target_and_lag_features(df_indicators)
    df_final = filter_neutral_rows(df_target)
    
    os.makedirs('data/processed', exist_ok=True)
    df_final.to_csv('data/processed/gold_processed_data.csv')
    print("Processed data saved to data/processed/gold_processed_data.csv")
    return df_final

if __name__ == "__main__":
    prepare_pipeline()
