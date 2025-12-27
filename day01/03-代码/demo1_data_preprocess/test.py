# coding:utf-8
# 准确语料库
# vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
# word2index = {value: i for i, value in enumerate(vocabs)}
# print(f'word2index-->{word2index}')
#
# for vocab in vocabs:
#     zero_list = [0]*len(vocabs)
#     idx = word2index[vocab]
#     zero_list[idx] = 1
#     print(f'当前单词{vocab}对应的one-hot是{zero_list}')
import torch.nn as nn
nn.Linear(2, 3)
