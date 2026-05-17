import pandas as pd
import numpy as np
import yfinance as yf
import os
from src.indicators import add_all_indicators

def download_data(ticker: str = 'GC=F', start_date: str = '2008-01-01', end_date: str = None) -> pd.DataFrame:
    """
    Downloads historical data from Yahoo Finance.
    Ensures at least 15 years of data for robust training after filtering.
    """
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Flatten multi-index columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.dropna(inplace=True)
    
    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/gold_raw_data.csv')
    print("Raw data saved to data/raw/gold_raw_data.csv")
    
    return df

def create_target_and_lag_features(df: pd.DataFrame, threshold: float = 0.003) -> pd.DataFrame:
    """
    Creates target variable based on FUTURE return (t+1) compared to features at day (t).
    Target: 1 (UP) if next day return > threshold
            0 (DOWN) if next day return < -threshold
            -1 (NEUTRAL) if between -threshold and threshold
    """
    df = df.copy()
    
    # Shift(-1) safely lags features: Row t predicts Return at t+1
    df['Next_Day_Return'] = df['Close'].pct_change().shift(-1)
    
    conditions = [
        (df['Next_Day_Return'] > threshold),
        (df['Next_Day_Return'] < -threshold)
    ]
    choices = [1, 0]
    df['Target'] = np.select(conditions, choices, default=-1) # -1 is NEUTRAL
    
    # Drop future info to prevent leakage
    df = df.drop(columns=['Next_Day_Return'])
    df = df.dropna(subset=['Target'])
    
    return df

def filter_neutral_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Removes NEUTRAL (-1) rows to reduce market noise."""
    initial_len = len(df)
    filtered_df = df[df['Target'] != -1].copy()
    final_len = len(filtered_df)
    print(f"Dropped {initial_len - final_len} NEUTRAL rows.")
    return filtered_df

def prepare_pipeline(ticker='GC=F'):
    """Full end-to-end preprocessing pipeline strictly adhering to order."""
    # 1. Download FULL continuous dataset
    df_raw = download_data(ticker)
    
    # 2. Compute ALL indicators on full data
    df_indicators = add_all_indicators(df_raw)
    
    # 3. Drop NaNs caused by rolling windows, then create lagged target
    df_indicators = df_indicators.dropna()
    df_target = create_target_and_lag_features(df_indicators)
    
    # 4. Remove NEUTRAL rows AFTER indicator calculation
    df_final = filter_neutral_rows(df_target)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    df_final.to_csv('data/processed/gold_processed_data.csv')
    print("Processed data saved to data/processed/gold_processed_data.csv")
    
    return df_final

if __name__ == "__main__":
    prepare_pipeline()
