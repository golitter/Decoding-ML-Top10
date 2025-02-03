import numpy as np

"""
 y = wx + b
"""


# print(dataset)
# print(x)
# print(y)
def get_w_b(inputx: np.array, inputy: np.array):
	x = np.array(inputx)
	y = np.array(inputy)
	# w = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean())**2).mean()
	w = ((y * (x - x.mean())).sum()) / ((x**2).sum() - (x.sum() ** 2) / len(x))
	b = y.mean() - w * x.mean()
	return w, b
