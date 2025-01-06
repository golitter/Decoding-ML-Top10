import numpy as np
import pandas as pd

def get_data():
    """
    x：以往成绩
    y：表现指数
    """
    data = pd.read_csv('./Student_Performance.csv', header=0)
    dataXY = np.array((data.iloc[0:,[1,5]].values))

    np.random.seed(114514)
    data_length = dataXY.shape[0]
    selected_data = np.random.choice(data_length, size=10, replace=False)
    dataXY = dataXY[selected_data]
    # print(dataXY)
    return dataXY[:,0], dataXY[:,1]

print(get_data())