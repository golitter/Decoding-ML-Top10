import numpy as np
import pandas as pd

def get_data():
    """
    x：以往成绩
    y：表现指数
    """
    data = pd.read_csv('./Student_Performance.csv', header=0)
    dataXY = np.array((data.iloc[0:,[1,5]].values))

    return dataXY[:,0], dataXY[:,1]

print(get_data())