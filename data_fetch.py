"""Download BTC-USD historical data and save to CSV.
Usage: python data_fetch.py --start 2020-01-01 --end 2024-12-31 --out data/bitcoin_price.csv
"""
import argparse
import os
import yfinance as yf
import pandas as pd

def fetch(start, end, out):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    df = yf.download("BTC-USD", start=start, end=end, interval="1d")
    df = df[['Open','High','Low','Close','Volume']]
    df.dropna(inplace=True)
    df.to_csv(out)
    print(f"Saved {len(df)} rows to {out}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2020-01-01')
    p.add_argument('--end', default='2024-12-31')
    p.add_argument('--out', default='data/bitcoin_price.csv')
    args = p.parse_args()
    fetch(args.start, args.end, args.out)
