import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.predict import get_latest_data, predict_next_day

# Configure page settings
st.set_page_config(page_title="Gold Trend Predictor", page_icon="📈", layout="wide")

# Header Section
st.title("🥇 Gold Trend Classification System")
st.markdown("""
Welcome to the **Gold Trend Predictor**. This application leverages a Machine Learning classifier to forecast whether Gold Futures (`GC=F`) will close **UP** or **DOWN** on the next trading day. 
*All technical indicators and features are calculated automatically in the background using the latest market data.*
""")

@st.cache_data(ttl=3600)
def load_and_predict():
    """Fetches data, calculates indicators, and runs inference."""
    df = get_latest_data()
    
    # STRICT RULE: Ensure we only use fully closed candles.
    # By dropping the last row, we guarantee the indicators are not corrupted by intra-day volatility.
    df_closed = df.iloc[:-1] 
    
    pred, prob = predict_next_day(df_closed)
    return df_closed, pred, prob

with st.spinner('Fetching market data and running ML model...'):
    df, pred, prob = load_and_predict()

st.header("🔮 Prediction for Next Trading Day")

col1, col2, col3 = st.columns(3)

# Format Prediction
trend = "UP 📈" if pred == 1 else "DOWN 📉"
confidence = prob[1] if pred == 1 else prob[0]

col1.metric("Predicted Direction", trend)
col2.metric("Model Confidence", f"{confidence*100:.2f}%")
col3.metric("Last Close Price", f"${df['Close'].iloc[-1]:.2f}")

st.divider()

st.header("📊 Market Data & Technical Indicators")

# Plotly Candlestick with SMAs
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Gold Price'))

fig.add_trace(go.Scatter(x=df.index, y=df['SMA_7'], line=dict(color='orange', width=1), name='SMA 7'))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_30'], line=dict(color='blue', width=1), name='SMA 30'))

fig.update_layout(title='Gold Futures (GC=F) - Recent Trend',
                  yaxis_title='Price (USD)',
                  template='plotly_dark',
                  xaxis_rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

# RSI Visualization
st.subheader("Relative Strength Index (RSI)")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI 14', line=dict(color='purple')))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
fig_rsi.update_layout(yaxis_title='RSI', template='plotly_dark', height=300)
st.plotly_chart(fig_rsi, use_container_width=True)

st.markdown("""
---
**Disclaimer**: This project is for educational and portfolio purposes only. Financial markets are highly unpredictable. Do not use this model for actual financial trading or investment decisions.
""")
