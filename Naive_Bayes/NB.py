from data_process import get_train_data, get_test_data
import numpy as np

# 朴素贝叶斯分类器


# 计算先验概率
def get_prior_prob(train_data: np.array):
	# 先验概率
	prior_prob = {}
	# 计算先验概率
	for i in range(len(train_data)):
		if train_data[i][-1] not in prior_prob:
			prior_prob[int(train_data[i][-1])] = 1
		else:
			prior_prob[int(train_data[i][-1])] += 1
	for key in prior_prob:
		prior_prob[key] /= len(train_data)
	return prior_prob


# 正太分布密度概率
def probability_density_function(mean: float, std: float, var: float) -> float:
	return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((var - mean) / std) ** 2)


# 计算条件概率
def get_cond_prob(train_data: np.array):
	n = len(train_data[0]) - 1
	cond_prob = [{} for i in range(n)]
	# 计算条件概率
	# (特征值, 类别) -> 出现次数
	fea_res_cnt = [[0, 0] for i in range(n)]
	for i in range(len(train_data)):
		for j in range(1, n):
			if int(train_data[i][-1]) == 0:
				fea_res_cnt[j][0] += 1
			else:
				fea_res_cnt[j][1] += 1

	for i in range(len(train_data)):
		# 仅计算特征值为离散值的条件概率
		for j in range(1, n - 2):
			fea = int(train_data[i][j])
			res = int(train_data[i][-1])
			if (fea, res) not in cond_prob[j]:
				cond_prob[j][(fea, res)] = 1
			else:
				cond_prob[j][(fea, res)] += 1

	# 计算特征值为连续值的条件概率
	reslist = [[[] for i in range(2)] for j in range(2)]
	for i in range(len(train_data)):
		for j in range(n - 2, n):
			res = int(train_data[i][-1])
			reslist[j - n + 2][res].append(float(train_data[i][j]))
	for i in range(2):
		for j in range(2):
			mean, std = np.mean(reslist[i][j]), np.std(reslist[i][j])
			cond_prob[i + n - 2][(j, mean, std)] = 0

	# 计算条件概率
	for i in range(1, n - 2):
		for key in cond_prob[i]:
			cond_prob[i][key] /= fea_res_cnt[i][key[1]]
	return cond_prob


# 测试
def test():
	cond_prob = get_cond_prob(get_train_data())
	prior_prob = get_prior_prob(get_train_data())
	test_data = get_test_data()
	# 预测
	right_cnt = 0
	for i in range(len(test_data)):
		good = bad = 1
		good = prior_prob[1]
		bad = prior_prob[0]
		for j in range(len(cond_prob)):
			for key in cond_prob[j]:
				if len(key) == 2:
					if key[1] == 0:
						bad *= cond_prob[j][(int(test_data[i][j]), key[1])]
					else:
						good *= cond_prob[j].get((int(test_data[i][j]), key[1]), 0)  # 有可能出现未知的特征值
				elif len(key) == 3:
					if key[0] == 0:
						bad *= probability_density_function(key[1], key[2], float(test_data[i][j]))
					else:
						good *= probability_density_function(key[1], key[2], float(test_data[i][j]))
		if good > bad:
			print('good')
			if int(test_data[i][-1]) == 1:
				print('right')
				right_cnt += 1
		else:
			print('bad')
			if int(test_data[i][-1]) == 0:
				print('right')
				right_cnt += 1

	print(f'accuracy: {right_cnt / len(test_data)}')


if __name__ == '__main__':
	# 获取训练集和测试集
	train_data = get_train_data()
	# 计算先验概率
	prior_prob = get_prior_prob(train_data)
	print(prior_prob)
	# 计算条件概率
	cond_prob = get_cond_prob(train_data)
	# print(cond_prob)
	# 查看条件概率
	for i in range(len(cond_prob)):
		print(f'特征{i}:')
		for key in cond_prob[i]:
			if len(key) == 2:
				print(f'P({key[0]} | {key[1]}): {cond_prob[i][key]}')
			elif len(key) == 3:
				print(f'(结果, mean, std) = ({key[0]} | {key[1]}, {key[2]})')
	test()
