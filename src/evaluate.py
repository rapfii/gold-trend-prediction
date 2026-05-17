import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.train import FEATURES, TARGET

# 🎨 STRICT GLOBAL VISUALIZATION SYSTEM
def setup_dark_theme():
    plt.style.use("dark_background")
    sns.set_theme(style="dark")
    plt.rcParams.update({
        'figure.facecolor': '#0E1117',
        'axes.facecolor': '#0E1117',
        'text.color': 'white',
        'axes.labelcolor': 'lightgray',
        'xtick.color': 'lightgray',
        'ytick.color': 'lightgray',
        'axes.edgecolor': '#333333'
    })

COLORS = {
    'blue': '#4FC3F7',
    'orange': '#FFB74D',
    'green': '#81C784',
    'red': '#E57373'
}

def evaluate_best_model():
    print("Loading data...")
    if not os.path.exists('data/processed/gold_processed_data.csv'):
        print("Processed data not found. Run 'python -m src.preprocess' first.")
        return
        
    df = pd.read_csv('data/processed/gold_processed_data.csv', index_col='Date', parse_dates=True)
    
    split_index = int(len(df) * 0.8)
    test_df = df.iloc[split_index:]
    
    X_test, y_test = test_df[FEATURES], test_df[TARGET]
    
    print("Loading model and scaler...")
    model = joblib.load('models/gold_trend_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, preds, target_names=['DOWN', 'UP']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    
    os.makedirs('images', exist_ok=True)
    setup_dark_theme()
    
    # 1. Heatmap
    plt.figure(figsize=(10, 8))
    corr = test_df[FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, 
                cbar_kws={'label': 'Correlation'}, 
                annot_kws={"color": "white"})
    plt.title('Feature Correlation Heatmap', color='white')
    plt.tight_layout()
    plt.savefig('images/heatmap.png', facecolor='#0E1117', bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'], annot_kws={"color": "white"})
    plt.title("Confusion Matrix", color='white')
    plt.ylabel('True Label', color='lightgray')
    plt.xlabel('Predicted Label', color='lightgray')
    plt.tight_layout()
    plt.savefig('images/confusion_matrix.png', facecolor='#0E1117', bbox_inches='tight')
    plt.close()
    
    # 3. Feature Importance
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        
    if importances is not None:
        indices = np.argsort(importances)
        plt.figure(figsize=(8, 6))
        plt.title('Feature Importances', color='white')
        plt.barh(range(len(indices)), importances[indices], color=COLORS['blue'], align='center')
        plt.yticks(range(len(indices)), [FEATURES[i] for i in indices])
        plt.xlabel('Relative Importance', color='lightgray')
        plt.tight_layout()
        plt.savefig('images/feature_importance.png', facecolor='#0E1117', bbox_inches='tight')
        plt.close()

    # 4. Price Trend (Recent 100 days from test set)
    recent_df = test_df.iloc[-100:]
    plt.figure(figsize=(10, 5))
    plt.plot(recent_df.index, recent_df['Close'], color='white', label='Close Price')
    plt.plot(recent_df.index, recent_df['SMA_7'], color=COLORS['orange'], label='SMA 7', alpha=0.8)
    plt.plot(recent_df.index, recent_df['SMA_30'], color=COLORS['blue'], label='SMA 30', alpha=0.8)
    plt.title('Gold Price Trend', color='white')
    plt.legend(facecolor='#0E1117', edgecolor='gray', labelcolor='white')
    plt.tight_layout()
    plt.savefig('images/price_trend.png', facecolor='#0E1117', bbox_inches='tight')
    plt.close()

    # 5. RSI Chart
    plt.figure(figsize=(10, 3))
    plt.plot(recent_df.index, recent_df['RSI_14'], color=COLORS['green'], label='RSI 14')
    plt.axhline(70, color=COLORS['red'], linestyle='--', alpha=0.5)
    plt.axhline(30, color=COLORS['blue'], linestyle='--', alpha=0.5)
    plt.title('Relative Strength Index', color='white')
    plt.legend(facecolor='#0E1117', edgecolor='gray', labelcolor='white')
    plt.tight_layout()
    plt.savefig('images/rsi_chart.png', facecolor='#0E1117', bbox_inches='tight')
    plt.close()

    # 6. MACD Chart
    plt.figure(figsize=(10, 3))
    plt.plot(recent_df.index, recent_df['MACD'], color=COLORS['blue'], label='MACD')
    plt.plot(recent_df.index, recent_df['MACD_Signal'], color=COLORS['orange'], label='Signal')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title('MACD Indicator', color='white')
    plt.legend(facecolor='#0E1117', edgecolor='gray', labelcolor='white')
    plt.tight_layout()
    plt.savefig('images/macd_chart.png', facecolor='#0E1117', bbox_inches='tight')
    plt.close()
        
    print("Dark theme evaluation visualizations saved to 'images/' directory.")

if __name__ == "__main__":
    evaluate_best_model()
