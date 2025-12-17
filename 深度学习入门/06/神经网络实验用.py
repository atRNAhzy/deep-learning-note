import numpy as np
import sys, os

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # 这会得到 /home/leaf/RoboticSoul/06/
parent_dir = os.path.dirname(current_dir)                # 这会得到 /home/leaf/RoboticSoul/

# 将项目根目录加入系统路径
sys.path.append(parent_dir)

# 现在可以正确导入
from dataset.mnist import load_mnist

import matplotlib.pyplot as plt
sys.path.append(os.pardir)
from common.layers import*
from common.gradient import numerical_gradient
from collections import OrderedDict
from optimizer import*
from common.optimizer import Adam

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

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


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


optimizer = AdaGrad()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ':' + str(diff))




# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1


train_loss_list = []
train_acc_list = []
test_acc_list = []
# 平均每个epoch的重复次数
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num +1):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grads = network.gradient(x_batch, t_batch)

    params = network.params
    # 更新参数
    optimizer.update(params, grads)



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