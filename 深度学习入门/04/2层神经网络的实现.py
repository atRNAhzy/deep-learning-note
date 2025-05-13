import numpy as np
import sys, os
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

#sigmoid函数的实现
def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

#softmax函数的实现
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

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


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        
        #初始化权重
        self.params={}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)
    

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y==t) / float(x.shape[0])

        return accuracy
    
    def gradient(self, x, t):
        # 前向传播
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # 反向传播
        dy = (y - t) / batch_size
        grad_W2 = np.dot(z1.T, dy)
        grad_b2 = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = da1 * sigmoid_grad(a1)
        grad_W1 = np.dot(x.T, dz1)
        grad_b1 = np.sum(dz1, axis=0)

        grads = {'W1': grad_W1, 'b1': grad_b1, 'W2': grad_W2, 'b2': grad_b2}
        return grads
    
# mini-batch 的实现
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

train_loss_list = []
train_acc_list = []
test_acc_list = []
# 平均每个epoch的重复次数
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'W2', 'b1', 'b2'):
        network.params[key] -= learning_rate * grad[key]



    # 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        # 记录学习过程
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"第{i}轮完成")
        print(f"train acc, test_acc |{train_acc:.2f}, {test_acc:.2f}" )

    
    

# 创建横轴（迭代次数）
iterations = range(len(train_loss_list))

# 绘制损失函数曲线
plt.plot(iterations, train_loss_list, label="Train Loss", color="black")

plt.plot(iterations, train_acc_list, label="Train_Accuracy", color="red")
plt.plot(iterations, test_acc_list, label="Test_Accuracy", color="blue")

# 添加标题和标签
plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")

# 显示图例
plt.legend()

# 显示图形
plt.show()
plt.savefig("training_loss_curve.png")