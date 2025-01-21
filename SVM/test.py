from data_processing import *
from smoSimple import *

data_mat_in, class_labels = data_processing()
b, alphas = smoSimple(data_mat_in, class_labels, 0.6, 0.001, 40)
print(b, alphas[alphas > 0])
