import numpy as np


def lda(X: np.array, y: np.array, n_components: int) -> np.array:
	"""
	LDA 降维
	"""
	# 获取类别列表
	classes = np.unique(y)
	n_features = X.shape[1]

	# 计算总均值
	mean_total = np.mean(X, axis=0)

	# 计算类内散度矩阵 Sw 和 类间散度矩阵 Sb
	S_W = np.zeros((n_features, n_features))
	S_B = np.zeros((n_features, n_features))

	for c in classes:
		X_c = X[y == c]  # 取出类别 c 的所有样本
		mean_c = np.mean(X_c, axis=0)  # 计算类别 c 的均值
		S_W += np.cov(X_c, rowvar=False) * (X_c.shape[0] - 1)  # 类内散度矩阵
		mean_diff = (mean_c - mean_total).reshape(-1, 1)
		S_B += X_c.shape[0] * (mean_diff @ mean_diff.T)  # 类间散度矩阵

	# 计算 Sw^-1 * Sb 的特征值和特征向量
	eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_W) @ S_B)

	# 选取前 n_components 个特征向量（按特征值降序排序）
	sorted_indices = np.argsort(eigvals)[::-1]
	W = eigvecs[:, sorted_indices[:n_components]]

	# 投影数据到 LDA 低维空间
	X_lda = X @ W

	return X_lda, W
