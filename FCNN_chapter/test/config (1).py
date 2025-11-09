"""超参数配置 - 方便调参"""


class Config:
    # 网络结构
    INPUT_SIZE = 784      # 28x28图像
    HIDDEN_SIZE = 128     
    OUTPUT_SIZE = 10      # 0-9数字
    
    # 训练设置
    LEARNING_RATE = 0.01
    EPOCHS = 100
    BATCH_SIZE = 64
    
    # 正则化
    REG_LAMBDA = 0.001    # L2正则化强度
    
    # 其他
    WEIGHT_INIT_STD = 0.01
    RANDOM_SEED = 42
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    
    # 日志
    VERBOSE = True
    SAVE_MODEL = True
    MODEL_SAVE_PATH = 'model_weights.npz'
