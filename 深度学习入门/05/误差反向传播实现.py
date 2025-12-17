import numpy as np


#softmax函数的实现
def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x-c) #通过常数c防止溢出
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
#交叉熵误差函数
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size



# 加法层的实现
class AddLayer:
    def __init__(self):
        pass
    def forward(self, x, y):
        out = x + y
        return out
    def backward(self, dout):
        dx = dout
        dy = dout
        return dx, dy
    
# 乘法层的实现
class MulLayer:
    def __init__(self):
        pass
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    def backward(self, dout):
        dx = self.y * dout
        dy = self.x * dout
        return dx, dy

# ReLU层的实现
class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

# Sigmoid层的实现
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

# Affine 层的实现（矩阵乘法）
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

# Softmax-with-Loss 层的实现(正规化输出)
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx






    
"""# 加法层测试
add_layer = AddLayer()

# 正向传播
x = 2.0
y = 3.0
out_add = add_layer.forward(x, y)
print("加法层正向传播结果:", out_add)  # 应该输出 5.0

# 反向传播
dout = 1.0  # 假设上游传来的梯度为1.0
dx, dy = add_layer.backward(dout)
print("加法层反向传播结果:", dx, dy)  # 应该输出 1.0, 1.0"""

"""# 乘法层测试
mul_layer = MulLayer()

# 正向传播
x = 2.0
y = 3.0
out_mul = mul_layer.forward(x, y)
print("乘法层正向传播结果:", out_mul)  # 应该输出 6.0

# 反向传播
dout = 1.0  # 假设上游传来的梯度为1.0
dx, dy = mul_layer.backward(dout)
print("乘法层反向传播结果:", dx, dy)  # 应该输出 3.0, 2.0"""

"""# 买苹果
apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 前向传播
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

# 反向传播
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)"""

