"""Compute common technical indicators and save features CSV.
Requires: data CSV with columns: Open, High, Low, Close, Volume
Outputs features including SMA, EMA, RSI, returns and lagged closes.
Usage: python feature_engineer.py --in data/bitcoin_price.csv --out data/bitcoin_features.csv
"""
import argparse, os
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator

def build_features(df):
    df = df.copy()
    df['SMA_7'] = SMAIndicator(df['Close'], window=7).sma_indicator()
    df['SMA_21'] = SMAIndicator(df['Close'], window=21).sma_indicator()
    df['EMA_12'] = EMAIndicator(df['Close'], window=12).ema_indicator()
    df['EMA_26'] = EMAIndicator(df['Close'], window=26).ema_indicator()
    df['RSI_14'] = RSIIndicator(df['Close'], window=14).rsi()
    df['Return'] = df['Close'].pct_change()
    df['LogReturn'] = (df['Close'] / df['Close'].shift(1)).apply(lambda x: 0 if pd.isna(x) else pd.np.log(x))
    # Lagged closes (for XGBoost)
    for lag in [1,2,3,7,14]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

def main(inp, outp):
    os.makedirs(os.path.dirname(outp) or '.', exist_ok=True)
    df = pd.read_csv(inp, parse_dates=True, index_col=0)
    df_feat = build_features(df)
    df_feat.to_csv(outp, index=True)
    print(f"Saved features to {outp} with {len(df_feat)} rows")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--out', dest='outp', default='data/bitcoin_features.csv')
    args = p.parse_args()
    main(args.inp, args.outp)
