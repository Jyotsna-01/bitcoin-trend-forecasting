"""Train XGBoost regressor on engineered features.
Usage: python train_xgboost.py --data data/bitcoin_features.csv --model_out models/xgb.json
"""
import argparse, os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def main(data_path, model_out):
    os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # features: engineered indicators + lagged closes
    features = ['SMA_7','SMA_21','EMA_12','EMA_26','RSI_14','Return','Close_lag_1','Close_lag_2','Close_lag_3','Close_lag_7']
    df = df.dropna()
    X = df[features].values
    y = df['Close'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=False)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print('XGBoost RMSE:', rmse)
    joblib.dump(model, model_out)
    print('Saved XGBoost model to', model_out)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--model_out', default='models/xgb.pkl')
    args = p.parse_args()
    main(args.data, args.model_out)
