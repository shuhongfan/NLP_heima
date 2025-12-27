# -*- coding:utf-8 -*-
import jieba
import pandas as pd
from itertools import chain

# 1.读取数据
train_data = pd.read_csv('./cn_data/train.tsv', sep='\t')
print(f'train_data--》{train_data}')
# 2. 获取句子长度分布
train_data['sentence_length'] =list(map(lambda x: len(x), train_data["sentence"]))
print(f'train_data--》{train_data}')
# 3. 获取训练集的所有单词的总数（去重）
train_vocab_size =set(chain(*map(lambda x: jieba.lcut(x), train_data["sentence"])))
print(f'train_vocab_size--》{len(train_vocab_size)}')


# 1.读取数据
dev_data = pd.read_csv('./cn_data/dev.tsv', sep='\t')
print(f'dev_data--》{dev_data}')
# 2. 获取训练集的所有单词的总数（去重）
dev_vocab_size = set(chain(*map(lambda x: jieba.lcut(x), dev_data["sentence"])))
print(f'dev_vocab_size--》{len(dev_vocab_size)}')
