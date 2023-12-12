# import torch
import sklearn.metrics as metrics
import numpy as np

# MAE, MSE, RMSE 추가
def MAE(output, target):
    return metrics.mean_absolute_error(target, output)

def MSE(output, target):
    return metrics.mean_squared_error(target, output)

def RMSE(output, target):
    return np.sqrt(metrics.mean_squared_error(target, output))



