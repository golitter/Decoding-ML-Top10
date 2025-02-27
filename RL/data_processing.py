import numpy as np


class KArmedBandit:
	def __init__(self, k=10, true_reward_mean=0, true_reward_std=1):
		"""
		k: 摇臂数量
		true_reward_mean: 奖励均值的均值
		true_reward_std: 奖励均值的标准差
		"""
		self.k = k
		self.q_true = np.random.normal(true_reward_mean, true_reward_std, k)  # 每个摇臂的真实均值

	def step(self, action):
		"""执行动作（拉某个摇臂），返回奖励"""
		reward = np.random.normal(self.q_true[action], 1)  # 以 q*(a) 为均值的正态分布
		return reward
