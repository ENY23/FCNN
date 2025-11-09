"""两层全连接神经网络"""

import numpy as np
from activation_functions import relu, relu_derivative, softmax
from loss_functions import cross_entropy_loss, cross_entropy_derivative


class TwoLayerNN:
    """简单的两层网络: Input -> Hidden(ReLU) -> Output(Softmax)"""
    
    def __init__(self, input_size, hidden_size, output_size,
                 learning_rate=0.01, reg_lambda=0.001, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        
        # He初始化适合ReLU
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Xavier初始化适合Softmax
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.cache = {}  # 存前向传播的中间结果
    
    def forward(self, X):
        """前向传播，返回预测概率"""
        # 隐藏层
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = relu(Z1)
        
        # 输出层
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = softmax(Z2)
        
        # 缓存给反向传播用
        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        
        return A2
    
    def backward(self, y_true):
        """反向传播计算梯度"""
        batch_size = y_true.shape[0]
        X = self.cache['X']
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        Z1 = self.cache['Z1']
        
        # 输出层梯度 (softmax + cross-entropy组合求导)
        dZ2 = cross_entropy_derivative(A2, y_true)
        dW2 = np.dot(A1.T, dZ2) / batch_size + self.reg_lambda * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size
        
        # 隐藏层梯度
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(X.T, dZ1) / batch_size + self.reg_lambda * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def update_parameters(self, gradients):
        """梯度下降更新权重"""
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']
    
    def compute_loss(self, y_pred, y_true):
        """损失 = 交叉熵 + L2正则"""
        data_loss = cross_entropy_loss(y_pred, y_true)
        reg_loss = 0.5 * self.reg_lambda * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return data_loss + reg_loss
    
    def predict(self, X):
        """预测类别（返回索引）"""
        return np.argmax(self.forward(X), axis=1)
    
    def compute_accuracy(self, X, y_true):
        """计算准确率"""
        predictions = self.predict(X)
        if y_true.ndim > 1:  # one-hot转索引
            y_true = np.argmax(y_true, axis=1)
        return np.mean(predictions == y_true)
    
    def save_weights(self, filepath):
        """保存权重到.npz文件"""
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"模型权重已保存到 {filepath}")
    
    def load_weights(self, filepath):
        """从.npz文件加载权重"""
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        print(f"模型权重已从 {filepath} 加载")
