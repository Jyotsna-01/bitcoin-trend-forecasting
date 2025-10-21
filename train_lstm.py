"""Train an LSTM on scaled Close prices (PyTorch).
This follows the approach in the project document (sequence length 60).
Usage: python train_lstm.py --data data/bitcoin_features.csv --model_out models/lstm.pth
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from evaluate import rmse, mae, plot_actual_predicted

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 50)
        c0 = torch.zeros(2, x.size(0), 50)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

def create_sequences(values, seq_len=60):
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len])
    return np.array(X), np.array(y)

def main(data_path, model_out):
    os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    close = df['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)
    SEQ = 60
    X, y = create_sequences(scaled, seq_len=SEQ)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

    model = LSTMModel(input_size=1)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    # evaluate
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    preds_rescaled = scaler.inverse_transform(preds)
    actual_rescaled = scaler.inverse_transform(y_test.reshape(-1,1))
    print('RMSE:', rmse(actual_rescaled, preds_rescaled))
    print('MAE:', mae(actual_rescaled, preds_rescaled))
    # save model and scaler
    torch.save({'model_state_dict': model.state_dict(), 'scaler_min': scaler.min_, 'scaler_scale': scaler.scale_}, model_out)
    print('Saved LSTM to', model_out)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--model_out', default='models/lstm.pth')
    args = p.parse_args()
    main(args.data, args.model_out)
