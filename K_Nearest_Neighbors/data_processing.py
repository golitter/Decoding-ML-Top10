import numpy as np
import pandas as pd


def pca(X: np.array, n_components: int) -> np.array:
	"""
	PCA 进行降维。
	"""
	# 1. 数据标准化（去均值）
	X_mean = np.mean(X, axis=0)
	X_centered = X - X_mean

	# 2. 计算协方差矩阵
	covariance_matrix = np.cov(X_centered, rowvar=False)

	# 3. 计算特征值和特征向量
	eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

	# 4. 按特征值降序排序
	sorted_indices = np.argsort(eigenvalues)[::-1]
	top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]

	# 5. 投影到新空间
	X_pca = np.dot(X_centered, top_eigenvectors)

	return X_pca


def get_data():
	data = pd.read_csv('iris.csv', header=None)
	# print(data.dtypes)
	unq = data.iloc[:, -1].unique()
	for i, u in enumerate(unq):
		data.iloc[:, -1] = data.iloc[:, -1].apply(lambda x: i if x == u else x)

	# print(data.sample(5))
	xuanze = np.random.choice([True, False], len(data), replace=True, p=[0.8, 0.2])
	train_data = data[xuanze]
	test_data = data[~xuanze]
	train_data = np.array(
		train_data,
		dtype=np.float32,
	)
	test_data = np.array(test_data, dtype=np.float32)
	# 归一化
	train_data[:, :-1] = (train_data[:, :-1] - train_data[:, :-1].mean(axis=0)) / train_data[:, :-1].std(axis=0)
	test_data[:, :-1] = (test_data[:, :-1] - test_data[:, :-1].mean(axis=0)) / test_data[:, :-1].std(axis=0)
	return (
		pca(train_data[:, :-1], 2),
		train_data[:, -1].astype(np.int32),
		pca(test_data[:, :-1], 2),
		test_data[:, -1].astype(np.int32),
	)


if __name__ == '__main__':
	x_train, y_train, x_test, y_test = get_data()
	print(y_train.dtype)
	print(x_test, y_test)
	print(x_train.shape, y_train.shape)
