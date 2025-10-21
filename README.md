# Bitcoin Trend Forecasting: Analyzing and Predicting Cryptocurrency Trends

Rebuilt project based on the provided mini project document.
This bundle contains scripts to download Bitcoin data, engineer features, train three models (ARIMA, XGBoost, LSTM),
evaluate them, and produce plots of actual vs predicted prices.

## Structure
- data_fetch.py          : Download BTC-USD historical data using yfinance and save CSV.
- feature_engineer.py    : Load CSV and compute technical indicators (SMA, EMA, RSI, returns).
- arima_baseline.py      : Train and evaluate ARIMA model (statsmodels).
- train_xgboost.py       : Train and evaluate XGBoost regressor using engineered features.
- train_lstm.py          : Train and evaluate LSTM (PyTorch) on scaled close prices.
- evaluate.py            : Utility evaluation functions (RMSE, MAE) and plotting helpers.
- requirements.txt       : Python dependencies.
- example_notebook.ipynb : Short walkthrough (skeleton) you can open in Jupyter.
- LICENSE                : MIT license.

## Quick start (recommended)
1. Create a virtual environment (python 3.9+ recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Fetch data:
   ```bash
   python data_fetch.py --start 2020-01-01 --end 2024-12-31 --out data/bitcoin_price.csv
   ```
3. Compute features:
   ```bash
   python feature_engineer.py --in data/bitcoin_price.csv --out data/bitcoin_features.csv
   ```
4. Train models and evaluate:
   ```bash
   python train_lstm.py --data data/bitcoin_features.csv --model_out models/lstm.pth
   python train_xgboost.py --data data/bitcoin_features.csv --model_out models/xgb.json
   python arima_baseline.py --data data/bitcoin_price.csv
   ```

## Notes
- The LSTM uses sequence length 60 by default (as in the original document).
- XGBoost uses feature windowed aggregates; ARIMA is provided as baseline.
- For production usage, consider adding hyperparameter tuning, model saving/loading utilities,
  more robust train/test splitting, cross validation and improved feature sets (on-chain metrics, sentiment).
