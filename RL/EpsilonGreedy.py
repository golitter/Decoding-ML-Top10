from data_processing import KArmedBandit
import numpy as np
import matplotlib.pyplot as plt


def select_action(epsilon: float, q_estimates: np.ndarray):
	"""根据 epsilon-greedy 策略选择动作"""
	if np.random.rand() < epsilon:  # 随机选择
		return np.random.choice(len(q_estimates))  #
	else:
		return np.argmax(q_estimates)  # 选择估计奖励最高的动作


def update_estimates(q_estimates: np.ndarray, action: int, reward: float, action_counts: np.ndarray):
	"""更新动作的估计奖励"""
	action_counts[action] += 1
	q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]
	return q_estimates, action_counts


def start(k: int, epsilon: float, epochs: int, stps: int):
	"""开始运行 epsilon-greedy 算法"""
	q_estimates = np.zeros(k)  # 每个摇臂的估计奖励
	action_counts = np.zeros(k)  # 每个摇臂被选择的次数
	avg_rewards = np.zeros(stps)  # 记录每次拉摇臂的奖励

	for epoch in range(epochs):
		bandit = KArmedBandit(k)
		rewards = []
		for step in range(stps):
			action = select_action(epsilon, q_estimates)
			reward = bandit.step(action)
			q_estimates, action_counts = update_estimates(q_estimates, action, reward, action_counts)
			rewards.append(reward)  # 记录奖励
		avg_rewards += np.array(rewards)  # 记录每次拉摇臂的奖励
	avg_rewards /= epochs
	return avg_rewards


if __name__ == '__main__':
	k = 10
	epsilon = 0.1
	epochs = 2000
	stps = 1000
	avg_rewards = start(k, epsilon, epochs, stps)
	plt.plot(avg_rewards)
	plt.xlabel('Steps')
	plt.ylabel('Average reward')
	plt.title('RL: epsilon-greedy Performance')
	plt.show()
