import numpy as np

# t 为 one-hot 形式
def cross_entropy_error_one_hot(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    print("\n=== One-hot 形式计算过程 ===")
    print("预测值 y:\n", y)
    print("标签 t (one-hot):\n", t)
    
    # 计算对数概率
    log_y = np.log(y + 1e-7)
    print("对数概率 log(y + 1e-7):\n", log_y)
    
    # 计算 t * log(y)
    t_log_y = t * log_y
    print("t * log(y + 1e-7):\n", t_log_y)
    
    # 计算交叉熵误差
    error = -np.sum(t_log_y) / batch_size
    print("交叉熵误差 (one-hot 形式):", error)
    
    return error

# t 为标签形式
def cross_entropy_error_labels(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    print("\n=== 标签形式计算过程 ===")
    print("预测值 y:\n", y)
    print("标签 t (标签形式):\n", t)
    
    # 提取正确类别的概率
    correct_probs = y[np.arange(batch_size), t]
    print("正确类别的概率 y[np.arange(batch_size), t]:\n", correct_probs)
    
    # 计算对数概率
    log_correct_probs = np.log(correct_probs + 1e-7)
    print("对数概率 log(y[np.arange(batch_size), t] + 1e-7):\n", log_correct_probs)
    
    # 计算交叉熵误差
    error = -np.sum(log_correct_probs) / batch_size
    print("交叉熵误差 (标签形式):", error)
    
    return error

