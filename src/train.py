import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Strictly historical features
FEATURES = [
    'Volume', 'SMA_7', 'SMA_30', 'EMA_14', 'RSI_14', 
    'MACD', 'MACD_Signal', 'Daily_Return', 'Volatility_14', 
    'Price_Range', 'Momentum_10'
]
TARGET = 'Target'

def train_models():
    """Trains models with STRICT chronological splitting and scaling."""
    print("Loading processed data...")
    if not os.path.exists('data/processed/gold_processed_data.csv'):
        print("Processed data not found. Run 'python -m src.preprocess' first.")
        return
        
    df = pd.read_csv('data/processed/gold_processed_data.csv', index_col='Date')
    
    # 5. Split data chronologically (NO SHUFFLE)
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]
    
    print("Scaling features...")
    scaler = StandardScaler()
    # STRICT RULE: Fit scaler ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='logloss')
    }
    
    best_model = None
    best_acc = 0
    best_name = ""
    
    print("Training models...")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    print(f"\nBest Model: {best_name} with Accuracy: {best_acc:.4f}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/gold_trend_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and scaler saved to 'models/' directory.")

if __name__ == "__main__":
    train_models()
