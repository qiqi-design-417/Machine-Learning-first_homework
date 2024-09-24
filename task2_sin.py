# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:43:44 2024

@author: lsq20
"""

import numpy as np  
import matplotlib.pyplot as plt  

# 初始化参数为小随机值  
a1 = np.random.rand() * 0.01  
a3 = np.random.rand() * 0.01  
a5 = np.random.rand() * 0.01  

# 定义泰勒级数展开的函数  
def taylor_series(x, a1, a3, a5):  
    return a1 * x +a3 * (x ** 3) / 6 + a5 * (x ** 5) / 120  

# 定义损失函数  
def loss_function(a1, a3, a5, x, y):  
    predictions = taylor_series(x, a1, a3, a5)  
    loss = np.mean((predictions - y) ** 2)  
    return loss  

# 计算损失函数对参数的梯度  
def compute_gradients(a1, a3, a5, x, y):  
    predictions = taylor_series(x, a1, a3, a5)  
    gradient_a1 = (1/len(x)) * np.sum((predictions - y) * x)  
    gradient_a3 = (1/len(x)) * np.sum((predictions - y) * (x ** 3) / 6)  
    gradient_a5 = (1/len(x)) * np.sum((predictions - y) * (x ** 5) / 120)  
    return gradient_a1, gradient_a3, gradient_a5  

# 设置学习率和迭代次数  
learning_rate = 0.001  
n_iterations = 70000  

# 准备数据进行拟合  
def data_init(n):  
    x = np.linspace(0, np.pi * 2, n)  
    y = np.sin(x)  
    return x, y  

# 数据加载  
x, y = data_init(100)  

# 执行梯度下降  
for iteration in range(n_iterations):  
    gradient_a1, gradient_a3, gradient_a5 = compute_gradients(a1, a3, a5, x, y)  
    a1 -= learning_rate * gradient_a1  
    a3 -= learning_rate * gradient_a3  
    a5 -= learning_rate * gradient_a5  
    if iteration % 1000 == 0:  
        current_loss = loss_function(a1, a3, a5, x, y)  
        print(f"Iteration {iteration}, Loss: {current_loss:.6f}, a1: {a1:.6f}, a3: {a3:.6f}, a5: {a5:.6f}")  

# 打印最佳参数  
print("最佳参数:", "a1 =", a1, "a3 =", a3, "a5 =", a5)  

# 绘制原样本点和预测函数曲线  
plt.scatter(x, y, label='Original Data', s=8)  
plt.plot(x, taylor_series(x, a1, a3, a5), label='Fit curve', color='red')  
plt.grid(True)  
plt.legend()  
plt.title('Taylor Series Fit to Sin Function')  
plt.xlabel('x')  
plt.ylabel('sin(x)')  
plt.show()