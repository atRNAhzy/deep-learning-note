# coding: utf-8
import os
import sys

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
# 指定要添加的目录路径
directory_path = "/home/leaf/RoboticSoul"

# 将目录添加到 sys.path
if directory_path not in sys.path:
    sys.path.append(directory_path)

from common.util import im2col
import numpy as np

x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape) #(9, 75)

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)