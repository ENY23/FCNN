"""损失函数和导数"""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """交叉熵损失，用于分类
    
    公式: -sum(y_true * log(y_pred)) / batch_size
    """
    epsilon = 1e-15  # 防止log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


def cross_entropy_derivative(y_pred, y_true):
    """交叉熵对预测值的导数
    
    当配合softmax使用时，导数简化为: y_pred - y_true
    """
    return y_pred - y_true


def mean_squared_error(y_pred, y_true):
    """均方误差，用于回归"""
    return np.mean((y_pred - y_true) ** 2)


def mse_derivative(y_pred, y_true):
    """MSE的导数"""
    return 2 * (y_pred - y_true) / y_true.shape[0]
