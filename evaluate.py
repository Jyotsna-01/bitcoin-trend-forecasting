import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def plot_actual_predicted(dates, actual, predicted, title='Actual vs Predicted'):
    plt.figure(figsize=(12,6))
    plt.plot(dates, actual, label='Actual')
    plt.plot(dates, predicted, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()
