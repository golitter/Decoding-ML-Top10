import numpy as np


def pca(X: np.array, n_components: int) -> np.array:
	"""
	PCA 降维。
	"""
	# 1. 数据标准化（去均值）
	X_mean = np.mean(X, axis=0)
	X_centered = X - X_mean  # (m, n)
	# print(f'X_centered.shape: {X_centered.shape}')
	# 2. 计算协方差矩阵
	covariance_matrix = np.cov(X_centered, rowvar=False)

	# 3. 计算特征值和特征向量
	eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

	# 4. 按特征值降序排序
	sorted_indices = np.argsort(eigenvalues)[::-1]
	top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]  # (n, n_components)
	# print(f'top_eigenvectors.shape: {top_eigenvectors.shape}')
	# 5. 投影到新空间
	X_pca = np.dot(X_centered, top_eigenvectors)  # (m, n_components)
	# print(f'X_pca.shape: {X_pca.shape}')
	return X_pca


if __name__ == '__main__':
	np.random.seed(0)
	X = np.random.rand(30, 6)
	n_components = 3
	X_pca = pca(X, n_components)
