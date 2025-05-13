import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

input_data = np.random.randn(1000, 100)  # 1000个数据
node_num = 100  # 各隐藏层的节点（神经元）数
hidden_layer_size = 5  # 隐藏层有5层

# 定义四种不同的初始值
initial_values = [1, 0.01, np.sqrt(1.0 / node_num), np.sqrt(2.0 / node_num)]

# 创建一个大的画布
plt.figure(figsize=(20, 10))

for idx, initial_value in enumerate(initial_values):
    activations = {}  # 激活值的结果保存在这里
    x = input_data

    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i-1]

        # 使用不同的初始值
        w = np.random.randn(node_num, node_num) * initial_value

        a = np.dot(x, w)

        # 使用Sigmoid激活函数
        z = sigmoid(a)

        activations[i] = z

    # 绘制直方图
    for i, a in activations.items():
        plt.subplot(len(initial_values), hidden_layer_size, idx * hidden_layer_size + i + 1)
        plt.title(f"Init: {initial_value}, Layer {i+1}")
        if i != 0: plt.yticks([], [])
        plt.hist(a.flatten(), 30, range=(0,1))

plt.tight_layout()
plt.savefig("不同初始值的激活值分布.png")
plt.show()