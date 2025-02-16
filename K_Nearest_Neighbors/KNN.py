from data_processing import get_data
import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(x_train: np.array, x_test: np.array) -> np.array:
	"""
	计算欧拉距离
	"""
	return np.sqrt(np.sum((x_train - x_test) ** 2, axis=1))


def knn(k: int, x_train: np.array, y_train: np.array, x_test: np.array) -> np.array:
	"""
	k近邻算法
	"""
	predictions = []
	for test in x_test:
		distances = euclidean_distance(x_train, test)
		nearest_indices = np.argsort(distances)[:k]  # 返回最近的k个点的索引
		nearest_labels = y_train[nearest_indices]  # 返回最近的k个点的标签
		prediction = np.argmax(np.bincount(nearest_labels))  # 返回最近的k个点中出现次数最多的标签
		predictions.append(prediction)
	return np.array(predictions)


def accuracy(predictions: np.array, y_test: np.array) -> float:
	"""
	计算准确率
	"""
	return np.sum(predictions == y_test) / len(y_test)


if __name__ == '__main__':
	k = 5
	x_train, y_train, x_test, y_test = get_data()
	predictions = knn(k, x_train, y_train, x_test)
	acc = accuracy(predictions, y_test)
	print(f'准确率为: {acc * 100:.2f}')

	# 绘制训练数据
	plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='viridis', marker='o', label='Train Data', alpha=0.7)

	# 绘制测试数据
	plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='coolwarm', marker='x', label='Test Data', alpha=0.7)

	# 绘制预测结果
	plt.scatter(
		x_test[:, 0],
		x_test[:, 1],
		c=predictions,
		cmap='coolwarm',
		marker='.',
		edgecolor='black',
		alpha=0.7,
		label='Predictions',
	)

	# 添加标题和标签
	plt.title('KNN Classification Results')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.legend()

	# 显示图形
	plt.show()
