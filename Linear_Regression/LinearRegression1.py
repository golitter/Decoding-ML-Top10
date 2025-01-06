import numpy as np
import pandas as pd

"""
 y = wx + b
"""

# print(dataset)
# print(x)
# print(y)
def get_w_b(inputx: np.array, inputy: np.array):
    x = np.array(inputx)
    y = np.array(inputy)
    w = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean())**2).mean()
    b = y.mean() - w * x.mean()
    return w, b

