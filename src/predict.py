import joblib
import pandas as pd
import yfinance as yf
from src.indicators import add_all_indicators
from src.train import FEATURES

def get_latest_data(ticker='GC=F'):
    """Fetches recent data and automatically calculates indicators for prediction."""
    # Fetch 60 days to satisfy longest rolling window (SMA_30)
    df = yf.download(ticker, period='60d')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    
    # Apply indicators internally - no manual input required
    df_ind = add_all_indicators(df)
    
    return df_ind

def predict_next_day(df_latest: pd.DataFrame):
    """Predicts next day's movement using the deployed model."""
    try:
        model = joblib.load('models/gold_trend_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
    except FileNotFoundError:
        print("Model or scaler not found. Please run 'python -m src.train' first.")
        return None, None
        
    # Get the latest row of features
    latest_features = df_latest.iloc[-1:][FEATURES]
    
    # Scale
    latest_scaled = scaler.transform(latest_features)
    
    # Predict
    pred = model.predict(latest_scaled)[0]
    prob = model.predict_proba(latest_scaled)[0]
    
    return pred, prob

if __name__ == "__main__":
    df = get_latest_data()
    # STRICT RULE: Ignore incomplete candle
    df_closed = df.iloc[:-1]
    
    pred, prob = predict_next_day(df_closed)
    
    if pred is not None:
        trend = "UP 📈" if pred == 1 else "DOWN 📉"
        confidence = prob[1] if pred == 1 else prob[0]
        
        print(f"Prediction for next trading day: {trend}")
        print(f"Confidence: {confidence*100:.2f}%")
