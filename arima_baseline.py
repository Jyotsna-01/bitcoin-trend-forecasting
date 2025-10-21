"""Basic ARIMA baseline using statsmodels. Fits on Close price series and forecasts.
Usage: python arima_baseline.py --data data/bitcoin_price.csv --order 5 1 0
"""
import argparse
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima.model import ARIMA
from evaluate import rmse, mae, plot_actual_predicted

def main(data_path, order=(5,1,0)):
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    series = df['Close']
    split = int(len(series)*0.8)
    train, test = series[:split], series[split:]
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    pred = model_fit.forecast(steps=len(test))
    print('ARIMA RMSE:', rmse(test.values.reshape(-1,1), pred.values.reshape(-1,1)))
    # plot
    plot_actual_predicted(test.index, test.values, pred.values, title='ARIMA: Actual vs Forecast')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--order', nargs=3, type=int, default=[5,1,0])
    args = p.parse_args()
    main(args.data, tuple(args.order))
