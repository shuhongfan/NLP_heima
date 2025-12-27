# coding:utf-8
from itertools import chain
import jieba
import pandas as pd


def get_vocabs():
    # 读数据
    train_data = pd.read_csv('./cn_data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./cn_data/dev.tsv', sep='\t')
    # 统计训练集语料中词汇的总个数
    train_vocabs = set(chain(*map(lambda x: jieba.lcut(x), train_data["sentence"])))
    print(f'train_vocabs-——>的总数量--》{len(train_vocabs)}')
    dev_vocabs = set(chain(*map(lambda x: jieba.lcut(x), dev_data["sentence"])))
    print(f'dev_vocabs-——>的总数量--》{len(dev_vocabs)}')

get_vocabs()