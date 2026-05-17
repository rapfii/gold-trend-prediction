import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.train import FEATURES, TARGET

def evaluate_best_model():
    print("Loading data...")
    if not os.path.exists('data/processed/gold_processed_data.csv'):
        print("Processed data not found. Run 'python src/preprocess.py' first.")
        return
        
    df = pd.read_csv('data/processed/gold_processed_data.csv', index_col='Date')
    
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
    
    plt.figure(figsize=(10, 8))
    corr = test_df[FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('images/heatmap.png')
    plt.close()
    
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('images/confusion_matrix.png')
    plt.close()
    
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        
    if importances is not None:
        indices = np.argsort(importances)
        plt.figure(figsize=(8, 6))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='#2980b9', align='center')
        plt.yticks(range(len(indices)), [FEATURES[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('images/feature_importance.png')
        plt.close()
        
    print("Evaluation visualizations saved to 'images/' directory.")

if __name__ == "__main__":
    evaluate_best_model()
