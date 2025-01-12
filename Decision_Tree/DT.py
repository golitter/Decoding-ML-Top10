from data_processing import create_dataset
import math

def createTree(dataset: list, labels: list, feature_labels: list):
    """
    创建决策树
    :param dataset: 数据集
    :param labels: 标签
    :param feature_labels: 已选择的特征标签
    :return: 决策树
    """
    class_list = [example[-1] for example in dataset]
    # 如果所有类标签相同，则返回该类标签
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果数据集中只有一个特征，返回出现次数最多的类标签
    if len(dataset[0]) == 1:
        return majority_count(class_list)
    
    # 选择最佳特征
    best_feature = choose_best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    feature_labels.append(best_feature_label)

    # 创建决策树
    my_tree = {best_feature_label: {}}
    # 删除已选择的特征
    del labels[best_feature]

    feature_values = [example[best_feature] for example in dataset]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = createTree(split_dataset(dataset, best_feature, value), sub_labels, feature_labels)
    return my_tree

def majority_count(class_list: list):
    """
    返回出现次数最多的类标签
    :param class_list: 类标签列表
    :return: 出现次数最多的类标签
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_class_count[0][0]

def choose_best_feature_to_split(dataset: list):
    """
    选择最佳特征
    :param dataset: 数据集
    :return: 最佳特征的索引
    """
    num_features = len(dataset[0]) - 1 # 最后一列为类标签
    base_entropy = calc_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1

    # 计算每个特征的信息增益  -- ID3算法
    for i in range(num_features):
        feature_list = [example[i] for example in dataset]
        unique_values = set(feature_list)
        new_entropy = 0.0
        for value in unique_values:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * calc_entropy(sub_dataset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def split_dataset(dataset: list, axis: int, value: int):
    """
    按照给定特征和值划分数据集
    :param dataset: 数据集
    :param axis: 特征索引
    :param value: 特征值
    :return: 划分后的数据集
    """
    return_dataset = []
    for feature_vec in dataset:
        if feature_vec[axis] == value:
            reduced_feature_vec = feature_vec[:axis]
            reduced_feature_vec.extend(feature_vec[axis + 1:])
            return_dataset.append(reduced_feature_vec)
    return return_dataset

def calc_entropy(dataset: list):
    """
    计算数据集的熵
    :param dataset: 数据集
    :return: 熵
    """
    num_entries = len(dataset)
    label_counts = {}
    for feature_vec in dataset:
        current_label = feature_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        entropy -= prob * math.log2(prob)
    return entropy

if __name__ == '__main__':
    dataset, labels = create_dataset()
    feature_labels = []
    my_tree = createTree(dataset, labels, feature_labels)
    print(my_tree)