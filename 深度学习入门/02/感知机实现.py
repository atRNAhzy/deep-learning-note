import numpy as np

def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(x*w) + b
    if tmp > 0:
        return 1
    else:
        return 0
    
#此处实现与书中不同,直接将AND的输出取反    
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(x*w) + b
    if tmp > 0:
        return 0
    else:
        return 1

def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(x*w) + b
    if tmp > 0:
        return 1
    else:
        return 0
    
#异或门使用双层感知机实现
def XOR(x1,x2):
    if NAND(x1,x2):
        if OR(x1,x2):
            return 1
    return 0


x1=0
x2=0
print(XOR(x1,x2))