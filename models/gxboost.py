# models/xgboost_model.py
# CSI 300 Quant Strategy — XGBoost Model
# David Lupinski | FH BFI Wien | 2026

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os

# --- Block 1: Load Data ---
def load_data(data_dir='data'):
    """
    Loads train and test data from CSV files created by pipeline.py.
    Identical to random_forest.py — same data, different model.
    """
    train = pd.read_csv(f'{data_dir}/train_data.csv')
    test  = pd.read_csv(f'{data_dir}/test_data.csv')

    features = ['z_momentum', 'z_mfi', 'z_turnover_rate',
                'z_roe', 'z_earnings_yield', 'composite_score']

    X_train = train[features]
    y_train = train['label']
    X_test  = test[features]
    y_test  = test['label']

    print(f"✅ Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    return X_train, X_test, y_train, y_test

# --- Block 2: Train Model ---
def train_model(X_train, y_train):
    """
    Trains an XGBoost Classifier on the training data.
    
    learning_rate=0.1: how much each tree corrects the previous
                       lower = more careful = less overfitting
    n_estimators=100:  100 sequential trees
    max_depth=3:       shallow trees → less overfitting
    eval_metric='logloss': measures prediction error during training
    random_state=42:   reproducibility
    """
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)
    print(f"✅ XGBoost trained on {X_train.shape[0]} rows")

    return model

# --- Block 3: Evaluate Model ---
def evaluate_model(model, X_test, y_test):
    """
    Evaluates XGBoost on unseen test data.
    Same structure as random_forest.py → direct comparison possible.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 Test Accuracy: {acc:.2%}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Underperform', 'Outperform']))

    return y_pred

# --- Block 4: Feature Importance ---
def plot_feature_importance(model, feature_names,
                            save_path='report/figures/xgb_feature_importance.png'):
    """
    XGBoost has its own built-in feature importance.
    Same structure as random_forest.py → direct comparison possible.
    """
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind='barh', ax=ax, color='darkorange')
    ax.set_title('XGBoost — Feature Importance')
    ax.set_xlabel('Importance Score')
    ax.axvline(x=1/len(feature_names), color='red',
               linestyle='--', label='Equal weight baseline')
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"✅ Feature importance chart saved: {save_path}")
    plt.show()

# --- Block 5: Save Predictions ---
def save_predictions(X_test, y_test, y_pred,
                     save_path='data/xgb_predictions.csv'):
    """
    Saves test data + true labels + XGBoost predictions to CSV.
    
    The backtest will load both rf_predictions.csv and xgb_predictions.csv
    to compare the two models side by side.
    """
    results = X_test.copy()
    results['y_true'] = y_test.values
    results['y_pred'] = y_pred

    results.to_csv(save_path, index=False)
    print(f"✅ Predictions saved: {save_path} ({len(results)} rows)")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, X_train.columns.tolist())
    save_predictions(X_test, y_test, y_pred)  # ← neu