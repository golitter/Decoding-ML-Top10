from data_processing import *
from smoSimple import *
import numpy as np
import matplotlib.pyplot as plt

# 数据处理和 SVM 训练
data_mat_in, class_labels = data_processing()
b, alphas = smoSimple(data_mat_in, class_labels, 0.6, 0.001, 40)

# 打印结果
print("Bias (b):", b)
print("Non-zero alphas:", alphas[alphas > 0])

# 打印数据形状
print("Shape of data_mat_in:", np.shape(data_mat_in))
print("Shape of class_labels:", np.shape(class_labels))

# 将 Y 转换为一维数组（如果它是二维的）
Y = class_labels
# 提取不同类别的索引
class_1_indices = np.where(Y == 1)[0]  # 类别为 1 的样本索引
class_2_indices = np.where(Y == -1)[0]  # 类别为 -1 的样本索引
X = data_mat_in

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(X[class_1_indices, 0], X[class_1_indices, 1], c='blue', label='Class 1', alpha=0.5)
plt.scatter(X[class_2_indices, 0], X[class_2_indices, 1], c='red', label='Class -1', alpha=0.5)

# 计算权重向量 w
w = np.dot((alphas * Y).T, X).flatten()
# print(f"w: {w}")
print("Shape of X:", X.shape)  # 应该是 (m, n)
print("Shape of Y:", Y.shape)  # 应该是 (m, 1)
print("Shape of alphas:", alphas.shape)  # 应该是 (m, 1)

# 绘制超平面
# 超平面方程：w[0] * x1 + w[1] * x2 + b = 0
# 解出 x2: x2 = -(w[0] * x1 + b) / w[1]
x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
x2 = -(w[0] * x1 + b) / w[1]
print(f"w_shape: {w.shape}")
# 绘制超平面
plt.plot(x1, x2, label='SVM Hyperplane', color='green', linewidth=2)

# 标出支持向量
support_vectors_indices = np.where(alphas > 0)[0]  # 找到所有支持向量的索引
plt.scatter(X[support_vectors_indices, 0], X[support_vectors_indices, 1], 
            facecolors='none', edgecolors='k', s=50, label='Support Vectors')

# 添加图例和标签
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Data with SVM Hyperplane')
plt.legend()

# 显示图形
plt.show()