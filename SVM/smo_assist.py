import random


def select_Jrandom(i: int, m: int) -> int:
	"""
	随机选择一个不等于 i 的整数
	"""
	j = i
	while j == i:
		j = int(random.uniform(0, m))
	return j


def clip_alpha(alpha_j: float, H: float, L: float) -> float:
	"""
	修剪 alpha_j
	"""
	if alpha_j > H:
		alpha_j = H
	if alpha_j < L:
		alpha_j = L
	return alpha_j
