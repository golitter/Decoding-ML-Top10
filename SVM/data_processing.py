import numpy as np
import pandas as pd

# https://zhuanlan.zhihu.com/p/350836534
def data_processing():
    data_csv = pd.read_csv('mouse_viral_study.csv')
    data_csv = data_csv.dropna()
    # print(data_csv)
    X = data_csv.iloc[:-1, 0:2].values
    # print(X)
    Y = data_csv.iloc[:-1, 2].map({0: -1, 1: 1}).values
    Y = Y.reshape(-1, 1)
    # print(Y.shape)
    return X, Y

# X, Y = data_processing()
# print(X)