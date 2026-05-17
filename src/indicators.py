import pandas as pd

def calculate_sma(df: pd.DataFrame, column: str = 'Close', window: int = 14) -> pd.Series:
    """Calculates Simple Moving Average."""
    return df[column].rolling(window=window).mean()

def calculate_ema(df: pd.DataFrame, column: str = 'Close', window: int = 14) -> pd.Series:
    """Calculates Exponential Moving Average."""
    return df[column].ewm(span=window, adjust=False).mean()

def calculate_rsi(df: pd.DataFrame, column: str = 'Close', window: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (Momentum)."""
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df: pd.DataFrame, column: str = 'Close', fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculates Moving Average Convergence Divergence."""
    ema_fast = calculate_ema(df, column, fast)
    ema_slow = calculate_ema(df, column, slow)
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_daily_return(df: pd.DataFrame, column: str = 'Close') -> pd.Series:
    """Calculates percentage change in daily close."""
    return df[column].pct_change()

def calculate_rolling_volatility(df: pd.DataFrame, column: str = 'Close', window: int = 14) -> pd.Series:
    """Calculates rolling volatility (std deviation of returns)."""
    daily_returns = calculate_daily_return(df, column)
    return daily_returns.rolling(window=window).std()

def calculate_price_range(df: pd.DataFrame) -> pd.Series:
    """Calculates intraday volatility (High - Low)."""
    return df['High'] - df['Low']

def calculate_momentum(df: pd.DataFrame, column: str = 'Close', window: int = 10) -> pd.Series:
    """Calculates n-day absolute price momentum."""
    return df[column].diff(window)

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all technical indicators to the dataset.
    STRICT RULE: This MUST be run on the FULL CONTINUOUS dataset 
    before any rows are dropped to preserve rolling window accuracy.
    """
    df = df.copy()
    
    df['SMA_7'] = calculate_sma(df, window=7)
    df['SMA_30'] = calculate_sma(df, window=30)
    df['EMA_14'] = calculate_ema(df, window=14)
    df['RSI_14'] = calculate_rsi(df, window=14)
    
    macd, signal = calculate_macd(df)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    
    df['Daily_Return'] = calculate_daily_return(df)
    df['Volatility_14'] = calculate_rolling_volatility(df, window=14)
    df['Price_Range'] = calculate_price_range(df)
    df['Momentum_10'] = calculate_momentum(df, window=10)
    
    return df
