import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.predict import get_latest_data, predict_next_day

# Configure page settings
st.set_page_config(page_title="Gold Trend Predictor", page_icon="📈", layout="wide")

# 🎨 STRICT GLOBAL VISUALIZATION SYSTEM (Plotly mapping)
DARK_BG = '#0E1117'
COLORS = {
    'blue': '#4FC3F7',
    'orange': '#FFB74D',
    'green': '#81C784',
    'red': '#E57373'
}

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
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name='Gold Price', increasing_line_color=COLORS['green'], decreasing_line_color=COLORS['red']), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['SMA_7'], line=dict(color=COLORS['orange'], width=1.5), name='SMA 7'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_30'], line=dict(color=COLORS['blue'], width=1.5), name='SMA 30'), row=1, col=1)

# RSI Subplot
fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI 14', line=dict(color=COLORS['blue'])), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color=COLORS['red'], row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color=COLORS['green'], row=2, col=1)

# Theme updates
fig.update_layout(
    title='Gold Futures (GC=F) - Recent Trend & Momentum',
    template='plotly_dark',
    plot_bgcolor=DARK_BG,
    paper_bgcolor=DARK_BG,
    xaxis_rangeslider_visible=False,
    height=600,
    margin=dict(l=0, r=0, t=40, b=0)
)
fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="RSI", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
---
**Disclaimer**: This project is for educational and portfolio purposes only. Financial markets are highly unpredictable. Do not use this model for actual financial trading or investment decisions.
""")
