import numpy as np
import matplotlib.pyplot as plt

#阶跃函数的实现
def step_function(x):
    y = x > 0
    return y.astype(np.int8)

#sigmoid函数的实现
def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

#恒等函数的实现(仅仅用于统一形式)
def identity_function(x):
    return x

#softmax函数的实现
#注意防止溢出
def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x-c) #通过常数c防止溢出
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# 生成测试数据
x = np.arange(-5.0, 5.0, 0.1)  # 从 -5 到 5，步长为 0.1

# 计算各激活函数的输出
y_step = step_function(x)
y_sigmoid = sigmoid(x)
y_identity = identity_function(x)
y_softmax = softmax(x)  # Softmax 需要特殊处理，因为它是一个概率分布

# 绘制图像
plt.figure(figsize=(12, 8))

# 阶跃函数
plt.subplot(2, 2, 1)
plt.plot(x, y_step, label="Step Function")
plt.title("Step Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()

# Sigmoid函数
plt.subplot(2, 2, 2)
plt.plot(x, y_sigmoid, label="Sigmoid Function", color="orange")
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()

# 恒等函数
plt.subplot(2, 2, 3)
plt.plot(x, y_identity, label="Identity Function", color="green")
plt.title("Identity Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()

# Softmax函数
plt.subplot(2, 2, 4)
plt.plot(x, y_softmax, label="Softmax Function", color="red")
plt.title("Softmax Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()

# 显示图像
plt.tight_layout()
plt.show()