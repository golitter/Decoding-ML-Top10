import pandas as pd
import numpy as np

# https://sci2s.ugr.es/keel/dataset.php?cod=182#inicio
df = pd.read_csv('banana.dat', header=None)


def get_data():
	return np.array(df.iloc[:, :2])
