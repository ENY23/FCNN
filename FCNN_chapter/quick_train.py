"""快速训练高准确率MNIST模型"""

import numpy as np
from neural_network import TwoLayerNN
from trainer import Trainer
from data_utils import one_hot_encode, train_test_split
import time


def quick_train_model():
    """训练97%+准确率的模型"""
    print("=" * 70)
    print("快速训练 - 超高准确率模式")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        from sklearn.datasets import fetch_openml
        
        print("\n加载MNIST...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
        X, y = mnist.data, mnist.target.astype(int)
        
        print(f"✓ 加载完成")
        
        # 用35K样本训练
        n_samples = 35000
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]
        
        print(f"使用 {n_samples} 个样本（超高准确率模式）")
        
        # 归一化
        X = X / 255.0
        y_onehot = one_hot_encode(y, 10)
        
        # 拆分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=0.2, random_seed=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_seed=42
        )
        
        print(f"训练集: {X_train.shape[0]}")
        print(f"验证集: {X_val.shape[0]}")
        print(f"测试集: {X_test.shape[0]}")
        
        # 创建大模型
        print("\n创建模型...")
        model = TwoLayerNN(
            input_size=784,
            hidden_size=300,  # 大网络
            output_size=10,
            learning_rate=0.12,
            reg_lambda=0.00003,  # 小正则化
            random_seed=42
        )
        
        print(f"  784 -> 300(ReLU) -> 10(Softmax)")
        print(f"  学习率: 0.12")
        
        # 训练30轮
        print("\n开始训练...")
        print("-" * 70)
        
        trainer = Trainer(model, verbose=True)
        history = trainer.train(
            X_train, y_train,
            X_val, y_val,
            epochs=30,
            batch_size=64,
            early_stopping_patience=8
        )
        
        # 评估
        print("\n" + "=" * 70)
        print("评估结果")
        print("=" * 70)
        
        test_results = trainer.evaluate(X_test, y_test)
        
        # 保存
        model_path = 'mnist_digit_recognizer.npz'
        model.save_weights(model_path)
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print(f"✓ 完成！耗时: {total_time:.1f}秒")
        print(f"✓ 已保存: {model_path}")
        print(f"✓ 测试准确率: {test_results['accuracy']*100:.1f}%")
        print("=" * 70)
        
        return model
        
    except ImportError as e:
        print(f"错误: {e}")
        print("请安装: pip install scikit-learn")
        return None
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    model = quick_train_model()
    
    if model is not None:
        print("\n现在运行 digit_recognizer.py 开始手写识别！")
