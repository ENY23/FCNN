"""激活函数和导数实现"""

import numpy as np
from scipy.special import expit  # 用scipy的sigmoid避免溢出


def sigmoid(x):
    """Sigmoid激活: 1 / (1 + e^-x)"""
    return expit(x)


def sigmoid_derivative(x):
    """Sigmoid导数: s(x) * (1 - s(x))"""
    return x * (1 - x)


def relu(x):
    """ReLU激活: max(0, x)"""
    return np.maximum(0, x)


def relu_derivative(x):
    """ReLU导数: x > 0 ? 1 : 0"""
    return (x > 0).astype(float)


def softmax(x):
    """Softmax: 把一组数转成概率分布，用于多分类输出层"""
    # 减去max防止exp溢出
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def tanh(x):
    """Tanh激活: (e^x - e^-x) / (e^x + e^-x)"""
    return np.tanh(x)


def tanh_derivative(x):
    """Tanh导数: 1 - tanh^2(x)"""
    return 1 - x ** 2
