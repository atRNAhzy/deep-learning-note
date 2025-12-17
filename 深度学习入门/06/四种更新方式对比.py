import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.mnist import load_mnist
from common.layers import *
from common.gradient import numerical_gradient
from common.optimizer import *
from collections import OrderedDict

# 超参数配置
iters_num = 10000
batch_size = 100
learning_rate = 0.01


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        
        # 初始化权重
        self.params={}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t) 
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y==t) / float(x.shape[0])

        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads


# 初始化四个独立的网络和优化器
networks = {
    'SGD': TwoLayerNet(784, 50, 10),
    'Momentum': TwoLayerNet(784, 50, 10),
    'AdaGrad': TwoLayerNet(784, 50, 10),
    'Adam': TwoLayerNet(784, 50, 10)
}

optimizers = {
    'SGD': SGD(lr=learning_rate),
    'Momentum': Momentum(lr=learning_rate),
    'AdaGrad': AdaGrad(lr=learning_rate),
    'Adam': Adam(lr=learning_rate)
}

# 训练数据加载
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 训练结果存储
train_loss_history = {name: [] for name in networks.keys()}

# 进行四种优化器的训练
for name in networks.keys():
    print(f"Training with {name}...")
    network = networks[name]
    optimizer = optimizers[name]
    
    for i in range(iters_num):
        # 获取mini-batch
        batch_mask = np.random.choice(x_train.shape[0], batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # 计算梯度并更新参数
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)
        
        # 记录损失（每100次迭代）
        if i % 100 == 0:
            loss = network.loss(x_batch, t_batch)
            train_loss_history[name].append(loss)
    
    print(f"{name} training completed.\n")

# 可视化设置
plt.figure(figsize=(12, 8))
colors = {'SGD': 'red', 'Momentum': 'blue', 'AdaGrad': 'green', 'Adam': 'orange'}

# 绘制训练损失曲线
for name in train_loss_history.keys():
    iterations = np.arange(len(train_loss_history[name])) * 100
    plt.plot(iterations, train_loss_history[name], 
             label=name, color=colors[name], alpha=0.8)

plt.title("Optimizer Comparison (Loss History)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("training_loss_curve.png")
plt.show()