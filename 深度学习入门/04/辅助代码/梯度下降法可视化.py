import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def f(x):
    return x**2 + 10 * np.sin(x)

# 定义数值梯度函数
def numerical_gradient(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

# 梯度下降函数
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []  # 用于记录x的变化
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        x_history.append(x)  # 记录x的值
    return x, x_history

# 设置初始值和参数
init_x = -2.5  # 初始值
step_num = 100  # 迭代次数
lr_values = [2]  # 不同的学习率

# 绘制结果
plt.figure(figsize=(10, 6))
x_vals = np.linspace(-10, 10, 400)
plt.plot(x_vals, f(x_vals), label="f(x) = x² + 10*sin(x)", color="blue")

# 对不同学习率进行梯度下降并绘制
for lr in lr_values:
    _, x_history = gradient_descent(f, init_x, lr=lr, step_num=step_num)
    plt.plot(x_history, f(np.array(x_history)), label=f"lr={lr}", marker="o", markersize=4)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gradient Descent with Different Learning Rates")
plt.legend()
plt.grid(True)
plt.show()