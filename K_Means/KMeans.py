from data_processing import get_data
import numpy as np
import matplotlib.pyplot as plt


# 初始化聚类中心
def init_centroids(data: np.array, k: int) -> np.array:
	return data[np.random.choice(data.shape[0], k, replace=False)]


# 欧拉距离
def euclidean_distance(x: np.array, y: np.array) -> float:
	return np.sqrt(np.sum(np.square(x - y)))


# 计算每个样本点到k个聚类中心的距离
def compute_distance(data: np.array, centroids: np.array) -> np.array:
	distance = np.zeros((data.shape[0], centroids.shape[0]))
	for i in range(centroids.shape[0]):
		distance[:, i] = np.apply_along_axis(euclidean_distance, 1, data, centroids[i])
	return distance


# KMeans算法
def kmeans(data: np.array, k: int, max_iter: int = 10):
	centroids = init_centroids(data, k)
	for i in range(max_iter):
		distance = compute_distance(data, centroids)
		# 每个样本点到k个聚类中心的距离最小值的索引
		labels = np.argmin(distance, axis=1)
		for j in range(k):
			centroids[j] = np.mean(data[labels == j], axis=0)
	return labels, centroids


if __name__ == '__main__':
	data = get_data()
	k = 3
	centroids = init_centroids(data, k)
	labels, centroids = kmeans(data, k)
	# print(labels.shape)
	plt.title('K-Means Clustering', fontsize=16)
	plt.xlabel('Feature 1', fontsize=14)
	plt.ylabel('Feature 2', fontsize=14)
	plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
	plt.show()
