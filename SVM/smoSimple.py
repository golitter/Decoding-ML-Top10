from smo_assist import (
    select_Jrandom, 
    clip_alpha)

import numpy as np
import pdb

def smoSimple(data_mat_in:np.ndarray, class_labels:np.ndarray, C:float, toler:float, max_iter:int):
    """
    data_mat_in: 数据集
    class_labels: 类别标签
    C: 松弛变量
    toler: 容错率
    max_iter: 最大迭代次数
    """
    b = 0; # 初始化b
    m, n = np.shape(data_mat_in) # m: 样本数, n: 特征数
    alphas = np.zeros((m, 1)) # 初始化alpha
    iter = 0 # 迭代次数
    while iter < max_iter:
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, class_labels).T @ (data_mat_in @ data_mat_in[i, :].T)) + b
            """
             (1 , m) * (m, n) * (n, 1) = (1, 1) = 标量
                再 加上 b 就是 f(x) 的值
            """
            Ei = fXi - float(class_labels[i])
            """
            Ei = f(x) - y 预测误差
            """
            if (
                # 第一种情况：样本被误分类，且权重可以增加
                ((class_labels[i] * Ei < -toler) # 预测误差与标签方向相反，且误差大于容忍度
                  and (alphas[i] < C)) # 当前权重小于正则化参数 C，可以增加权重
                or 
                # 第二种情况：样本被误分类，且权重需要调整
                 ((class_labels[i] * Ei > toler) # 预测误差与标签方向相同，且误差大于容忍度
                   and (alphas[i] > 0)) # 当前权重大于 0，需要调整权重
                ):
                j = select_Jrandom(i, m)
                fxj = float(np.multiply(alphas, class_labels).T @ (data_mat_in @ data_mat_in[j, :].T)) + b
                Ej = fxj - float(class_labels[j])
                alpha_j_old = alphas[j].copy(); 
                alpha_i_old = alphas[i].copy()

                if (class_labels[i] != class_labels[j]):
                    L = max(0, alphas[j] - alphas[i]) # 左边界
                    H = min(C, C + alphas[j] - alphas[i]) # 右边界
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: 
                    continue # 跳出本次循环
                
                eta = 2.0 * data_mat_in[i, :] @ data_mat_in[j, :].T - data_mat_in[i, :] @ data_mat_in[i, :].T - data_mat_in[j, :] @ data_mat_in[j, :].T
                """
                计算 eta = K11 + K22 - 2 * K12 = 2 * x_i * x_j - x_i * x_i - x_j * x_j 
                """     
                if eta >= 0:
                    continue
                alphas[j] -= class_labels[j] * (Ei - Ej) / eta # 更新权重
                alphas[j] = clip_alpha(alphas[j], H, L) # 调整权重
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    continue # 跳出本次循环，不更新 i
                alphas[i] += class_labels[j] * class_labels[i] * (alpha_j_old - alphas[j]) # 更新权重
                b1 = b - Ei - class_labels[i] * (alphas[i] - alpha_i_old) * data_mat_in[i, :] @ data_mat_in[i, :].T - class_labels[j] *(alphas[j] - alpha_j_old) * data_mat_in[i, :] @ data_mat_in[j, :].T
                b2 = b - Ej - class_labels[i] * (alphas[i] - alpha_i_old) * data_mat_in[i, :] @ data_mat_in[j, :].T - class_labels[j] *(alphas[j] - alpha_j_old) * data_mat_in[j, :] @ data_mat_in[j, :].T
                """
                更新 b
                """     
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
    return b, alphas



if __name__ == '__main__':
    print(  smoSimple(np.array([[1, 2], [3, 4]]), np.array([[-1],[1]]), 0.6, 0.001, 40))