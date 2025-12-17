import numpy as np

# 均方差函数
def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)*2)

#交叉熵误差函数
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

print("1")