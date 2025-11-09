"""MNIST手写数字识别演示 - 完整训练流程"""

import numpy as np
from neural_network import TwoLayerNN
from trainer import Trainer
from data_utils import one_hot_encode, train_test_split


def load_mnist_data(n_samples=10000, random_seed=42):
    """加载MNIST数据集"""
    try:
        from sklearn.datasets import fetch_openml
        
        print("正在加载MNIST数据集...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
        X, y = mnist.data, mnist.target.astype(int)
        
        # 随机采样
        np.random.seed(random_seed)
        indices = np.random.choice(len(X), n_samples, replace=False)
        X, y = X[indices], y[indices]
        
        print(f"✓ 成功加载 {n_samples} 个样本")
        return X, y
        
    except ImportError:
        print("❌ 错误：未安装 scikit-learn")
        print("请运行: pip install scikit-learn")
        return None, None
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return None, None


def main():

    print("=" * 60)
    print("两层全连接神经网络 - MNIST演示")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n步骤 1: 加载MNIST")
    print("-" * 60)
    
    X, y = load_mnist_data(n_samples=10000, random_seed=42)
    
    if X is None or y is None:
        print("加载失败")
        return None, None, None
    
    print(f"数据: X={X.shape}, y={y.shape}")
    print(f"类别分布: {np.bincount(y)}")
    
    # 2. 预处理
    print("\n步骤 2: 预处理")
    print("-" * 60)
    
    X = X / 255.0  # 归一化到[0,1]
    print("✓ 像素归一化")
    
    y_onehot = one_hot_encode(y, num_classes=10)
    print(f"✓ One-hot编码: {y_onehot.shape}")
    
    # 拆分数据集 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_onehot, test_size=0.3, random_seed=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_seed=42
    )
    
    print(f"✓ 训练集: {X_train.shape[0]}")
    print(f"✓ 验证集: {X_val.shape[0]}")
    print(f"✓ 测试集: {X_test.shape[0]}")
    
    # 3. 创建模型
    print("\n步骤 3: 创建模型")
    print("-" * 60)
    
    model = TwoLayerNN(
        input_size=784,      # 28x28
        hidden_size=128,
        output_size=10,      # 0-9
        learning_rate=0.1,
        reg_lambda=0.001,
        random_seed=42
    )
    
    print(f"架构: 784 -> 128(ReLU) -> 10(Softmax)")
    print(f"学习率: {model.learning_rate}")
    print(f"L2正则: {model.reg_lambda}")
    
    # 4. 训练
    print("\n步骤 4: 训练")
    print("-" * 60)
    
    trainer = Trainer(model, verbose=True)
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        batch_size=64,
        early_stopping_patience=5
    )
    
    # 5. 评估
    print("\n步骤 5: 测试集评估")
    print("-" * 60)
    test_results = trainer.evaluate(X_test, y_test)
    
    # 6. 保存
    print("\n步骤 6: 保存模型")
    print("-" * 60)
    model_path = 'model_weights.npz'
    model.save_weights(model_path)
    
    # 7. 预测示例
    print("\n步骤 7: 预测示例")
    print("-" * 60)
    
    sample_indices = np.random.choice(X_test.shape[0], 5, replace=False)
    
    print(f"{'No.':<4} {'真实':<4} {'预测':<4} {'置信度':<8} {'结果':<4}")
    print("-" * 30)
    
    for i, idx in enumerate(sample_indices, 1):
        X_sample = X_test[idx:idx+1]
        y_true = np.argmax(y_test[idx])
        y_pred = model.predict(X_sample)[0]
        confidence = model.forward(X_sample)[0][y_pred]
        result = "✓" if y_pred == y_true else "✗"
        
        print(f"{i:<4} {y_true:<4} {y_pred:<4} {confidence:<8.1%} {result:<4}")
    
    print("\n" + "=" * 60)
    print(f"✓ 完成！测试准确率: {test_results['accuracy']*100:.2f}%")
    print("=" * 60)
    print("\n提示:")
    print("  • 高准确率训练: python quick_train.py")
    print("  • 手写识别GUI: python digit_recognizer.py")
    print("=" * 60)
    
    return model, history, test_results


if __name__ == "__main__":
    model, history, results = main()
    
    if model is not None:
        # 想画曲线的话取消注释:
        # trainer = Trainer(model)
        # trainer.history = history
        # trainer.plot_history()
        pass
