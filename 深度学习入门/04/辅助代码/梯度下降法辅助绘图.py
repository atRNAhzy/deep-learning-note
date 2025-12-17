import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def f(x):
    return x**2 + 10 * np.sin(x)

# 生成数据
x = np.linspace(-10, 10, 400)
y = f(x)

# 绘制函数图像
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="f(x) = x² + 10*sin(x)", color="blue")

# 设置标题和标签
plt.title("Single Variable Function with Multiple Minima")
plt.xlabel("x")
plt.ylabel("f(x)")

# 显示网格
plt.grid(True)

# 显示图例
plt.legend()

# 显示图形
plt.show()