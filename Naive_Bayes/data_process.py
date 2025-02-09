import pandas as pd
import numpy as np
from io import StringIO

data = '编号,色泽,根蒂,敲声,纹理,脐部,触感,密度,含糖率,好瓜\n\
1,青绿,蜷缩,浊响,清晰,凹陷,硬滑,0.697,0.46,是\n\
2,乌黑,蜷缩,沉闷,清晰,凹陷,硬滑,0.774,0.376,是\n\
3,乌黑,蜷缩,浊响,清晰,凹陷,硬滑,0.634,0.264,是\n\
4,青绿,蜷缩,沉闷,清晰,凹陷,硬滑,0.608,0.318,是\n\
5,浅白,蜷缩,浊响,清晰,凹陷,硬滑,0.556,0.215,是\n\
6,青绿,稍蜷,浊响,清晰,稍凹,软粘,0.403,0.237,是\n\
7,乌黑,稍蜷,浊响,稍糊,稍凹,软粘,0.481,0.149,是\n\
8,乌黑,稍蜷,浊响,清晰,稍凹,硬滑,0.437,0.211,是\n\
9,乌黑,稍蜷,沉闷,稍糊,稍凹,硬滑,0.666,0.091,否\n\
10,青绿,硬挺,清脆,清晰,平坦,软粘,0.243,0.267,否\n\
11,浅白,硬挺,清脆,模糊,平坦,硬滑,0.245,0.057,否\n\
12,浅白,蜷缩,浊响,模糊,平坦,软粘,0.343,0.099,否\n\
13,青绿,稍蜷,浊响,稍糊,凹陷,硬滑,0.639,0.161,否\n\
14,浅白,稍蜷,沉闷,稍糊,凹陷,硬滑,0.657,0.198,否\n\
15,乌黑,稍蜷,浊响,清晰,稍凹,软粘,0.36,0.37,否\n\
16,浅白,蜷缩,浊响,模糊,平坦,硬滑,0.593,0.042,否\n\
17,青绿,蜷缩,沉闷,稍糊,稍凹,硬滑,0.719,0.103,否'

df = pd.read_csv(StringIO(data))
# print(df.info())


def obj_to_int(series: pd.Series):
	return pd.Categorical(series).codes


def label_encoder(df: pd.DataFrame):
	for col in df.columns:
		if df[col].dtype == 'object':
			df[col] = obj_to_int(df[col])
	return df


# 已完成：2025年2月9日 14点27分
# # 保存处理后的数据
# df = label_encoder(df)
# df.to_csv('data.csv', index=False)

# # 已完成：2025年2月9日 14点30分
# # 随机选择三行作为测试集
# test_set = df.sample(n=3, random_state=42)
# # 获取剩余的行作为训练集
# train_set = df.drop(test_set.index)
# # 保存训练集和测试集
# train_set.to_csv('train.csv', index=False)
# test_set.to_csv('test.csv', index=False)


def get_train_data():
	df = pd.read_csv('train.csv')
	return np.array(df.iloc[:, :])


def get_test_data():
	# 2025年2月9日 16点19分
	df = pd.read_csv('test.csv')
	return np.array(df.iloc[0:, :])

	# 随机选择一半的数据作为训练集
	# df = pd.read_csv('data.csv')
	# bool_array = np.random.choice([True, False], size=len(df), p=[0.5, 0.5])
	# return np.array(df[bool_array].iloc[:, :])


# print(get_train_data())
# print(get_test_data())
