import numpy as np
import matplotlib.pyplot as plt
import data_processing as dp


def sigmoid(x: np.array):
	return 1 / (1 + np.exp(-x))


def trains(X: np.array, Y: np.array, theta: np.array, learn_rate: float, epochs: int):
	"""
	定义trains函数，用于训练逻辑回归模型
	X: np.array，特征矩阵，形状为(m, n)，其中m是样本数量，n是特征数量
	Y: np.array，目标变量，形状为(m, 1)，表示每个样本的标签（0或1）
	theta: np.array，模型参数，形状为(n, 1)，表示每个特征的权重
	learn_rate: float，学习率，控制参数更新的步长
	epochs: int，迭代次数，表示训练模型的轮数
	"""
	m = len(Y)

	loss_values = np.zeros((epochs, 1))
	# 进行多次迭代训练
	for epoch in range(epochs):
		z = X @ theta
		# 计算预测值h，通过sigmoid函数将z映射到(0, 1)之间
		h = sigmoid(z)
		# 计算损失loss，即预测值h与实际值Y之间的差异
		loss = h - Y
		# 计算梯度gradient，根据损失和特征矩阵计算参数的梯度
		gradient = X.T @ loss / m
		# 更新参数theta，使用梯度下降法，学习率乘以梯度
		theta -= learn_rate * gradient
		loss_values[epoch] = np.sum(loss**2) / (2 * m)

	plt.plot(loss_values)
	plt.show()
	# 返回训练后的参数theta
	return theta


X, Y = dp.data_processing()
print(trains(X, Y, np.zeros((X.shape[1], 1)), 0.01, 10000))
