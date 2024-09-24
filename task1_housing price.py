# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:32:02 2024

@author: lsq20
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
house = pd.read_csv("D:\\Program Files\\machine learning\\boston.csv", header=0)

# 对特征数据【0到13】列做（0-1）归一化
for i in range(14):  
    house.iloc[:, i] = (house.iloc[:, i] - house.iloc[:, i].min()) / (house.iloc[:, i].max() - house.iloc[:, i].min())

# 前13列数据为特征数据，最后一列平均房价为标签数据
x_data = house.iloc[:, :13].values
y_data = house.iloc[:, 13].values

# 添加偏置项
x_data = np.c_[np.ones(x_data.shape[0]), x_data]

# 计算线性回归参数（最小二乘法）
def linear_regression(x, y):
    x_transpose = x.T
    coefficients = np.linalg.inv(x_transpose.dot(x)).dot(x_transpose).dot(y)
    return coefficients

coefficients = linear_regression(x_data, y_data)

# 预测
def predict(x, coefficients):
    return x.dot(coefficients)

predictions = predict(x_data, coefficients)

# 绘制原始样本点和预测函数曲线
plt.scatter(y_data, predictions,s=10)
plt.xlabel('Real housing prices')
plt.ylabel('Predicting housing prices')
plt.title('Real vs Predicting')

# 绘制y=x参考线
min_value = min(y_data.min(), np.min(predictions))
max_value = max(y_data.max(), np.max(predictions))
plt.plot([min_value, max_value], [min_value, max_value], 'k--', lw=2, color='red')
plt.show()
