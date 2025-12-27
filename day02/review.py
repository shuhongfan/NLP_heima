# a = [0]*6
# print(a)

# vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
#
# for idx, value in enumerate(vocabs):
#     zero_list = [0] * len(vocabs)
#     # 找到当前单词对应的索引
#     zero_list[idx] = 1
#     print(f'当前单词{value}的one-hot编码是{zero_list}')

# a = {word: [1 if j == i else 0 for j in range(len(vocabs))] for i, word in enumerate(vocabs)}
# print(a)
#
# import pandas as pd
# vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
# df = pd.DataFrame(vocabs)
# print(df.head())
# df2 = pd.get_dummies(df)
# print(df2)
# df2 = df2.astype(int)
# print(df2.values.tolist())
# import numpy as np
# a = np.array([[1, 2, 3], [4, 5,6]])
# print(a.ndim)
#
# import torch
# b = torch.tensor([[4]])
# print(b.shape)
# print(b.size())

def fun(a):
    return a


print(fun("niaho "))