import sys
sys.path.append('../')

import numpy as np
from utils.features import prepare_for_training

class LinearRegression:
	# data: 输入数据集, m * n 的矩阵, m 为样本数量, n 为特征数量
	# labels: 输入数据集的标签， m * 1 的矩阵, 监督学习的标签

	def __init__(self, data, labels, polynomial_degree = 0, sinusoid_degree = 0, normalize_data = True):
		(data_processed, features_mean, features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)
		self.data = data_processed
		self.labels = labels
		self.features_mean = features_mean
		self.features_deviation = features_deviation
		self.polynomial_degree = polynomial_degree
		self.sinusoid_degree = sinusoid_degree
		self.normalize_data = normalize_data
		self.errors = []

		num_features = self.data.shape[1] # 矩阵的列数, 即特征的数量, n
		self.theta = np.zeros((num_features, 1)) # 对应文档中的 h_θ

	def train(self, alpha, num_iterations = 500):
		self.gradient_descent(alpha, num_iterations)
		return self.theta, self.errors
	
	def gradient_descent(self, alpha, num_iterations):
		num_examples = self.data.shape[0]
		# 批量梯度下降, 每次迭代都要计算所有样本的梯度
		for i in range(num_iterations):
			predictions = np.dot(self.data, self.theta) # 预测值，矩阵相乘 data (m, n) * theta (n, 1) = predictions (m, 1), 当有n个特征时，为n元1次方程 y = n1x1 + n2x2 + ... + nnxn
			errors = predictions - self.labels # loss = 预测值 - 真实值, 矩阵相减 predictions (m, 1) - labels (m, 1) = errors (m, 1)
			self.errors.append(np.sum(errors ** 2) / (2 * num_examples)) # 计算损失函数，(errors^2) / (2 * m)
			gradient = np.dot(self.data.T, errors) / num_examples # 梯度 = data.T (n, m) * errors (m, 1) = gradient (n, 1), 矩阵乘积已经包含了求和
			self.theta -= alpha * gradient # 更新 theta，矩阵相减 theta (n, 1) - alpha * gradient (n, 1) = theta (n, 1)
	
	def predict(self, data):
		(data_processed, _, _) = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)
		predictions = np.dot(data_processed, self.theta)
		return predictions


# 矩阵乘法例子 (5,2) * (2,3) = (5,3)
# 1 2   1 2 3   1*1+2*4 1*2+2*5 1*3+2*6   9  12 15
# 3 4 * 4 5 6 = 3*1+4*4 3*2+4*5 3*3+4*6   19 26 33
# 5 6           5*1+6*4 5*2+6*5 5*3+6*6 = 29 40 51
# 7 8           7*1+8*4 7*2+8*5 7*3+8*6   39 54 69
# 9 0           9*1+0*4 9*2+0*5 9*3+0*6   9  18 27