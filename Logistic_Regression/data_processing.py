import pandas as pd


def data_processing():
	data_csv = pd.read_csv('iris.data')
	data_csv = data_csv.dropna()
	# print(data_csv)
	X = data_csv.iloc[:-1, 0:4].values
	# print(X)
	Y = data_csv.iloc[:-1, 4].map({'Iris-setosa': 0, 'Iris-versicolor': 1}).values
	Y = Y.reshape(-1, 1)
	# print(Y.shape)
	return X, Y


data_processing()
