"""数据处理工具"""

import numpy as np


def one_hot_encode(labels, num_classes):
    """类别标签转one-hot编码
    
    例: [0, 1, 2] -> [[1,0,0], [0,1,0], [0,0,1]]
    """
    n_samples = labels.shape[0]
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), labels] = 1
    return one_hot


def normalize_data(X, method='standardize'):
    """数据归一化
    
    method='standardize': (X - mean) / std  标准化到均值0方差1
    method='minmax': (X - min) / (max - min)  缩放到[0, 1]
    """
    if method == 'standardize':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8  # 防止除0
        X_normalized = (X - mean) / std
        params = {'mean': mean, 'std': std}
    
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}
    
    else:
        raise ValueError(f"未知方法: {method}")
    
    return X_normalized, params


def apply_normalization(X, params, method='standardize'):
    """用已有参数归一化新数据"""
    if method == 'standardize':
        return (X - params['mean']) / params['std']
    elif method == 'minmax':
        return (X - params['min']) / (params['max'] - params['min'])


def train_test_split(X, y, test_size=0.2, random_seed=None):
    """拆分训练集和测试集"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def create_mini_batches(X, y, batch_size, shuffle=True):
    """生成小批量数据"""
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def shuffle_data(X, y, random_seed=None):
    """打乱数据顺序"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    return X[indices], y[indices]
