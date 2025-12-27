# coding:utf-8
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def dm_content_length():
    plt.style.use('fivethirtyeight')
    # # 读数据
    # train_data = pd.read_csv('./cn_data/train.tsv', sep='\t')
    # # 添加一个句子长度列
    # train_data["sentence_length"] = list(map(lambda x: len(x), train_data["sentence"]))
    # print(f'train_data--》{train_data.head()}')
    # # 画图:柱状图
    # sns.countplot(x='sentence_length', data=train_data)
    # # plt.xticks([])
    # plt.title('train_data')
    # plt.show()
    # # 画图：曲线图
    # sns.displot(x='sentence_length', data=train_data, kind='kde')
    # # plt.xticks([])
    # plt.title('train_data')
    # plt.show()
    # 验证集
    dev_data = pd.read_csv('./cn_data/dev.tsv', sep='\t')
    # 添加一个句子长度列
    dev_data["sentence_length"] = list(map(lambda x: len(x), dev_data["sentence"]))
    print(f'dev_data--》{dev_data.head()}')
    # 画图:柱状图
    sns.countplot(x='sentence_length', data=dev_data)
    # plt.xticks([])
    plt.title('dev_data')
    plt.show()
    # 画图：曲线图
    sns.displot(x='sentence_length', data=dev_data, kind='kde')
    # plt.xticks([])
    plt.title('dev_data')
    plt.show()


def dm_stripplot():
    # 读数据
    train_data = pd.read_csv('./cn_data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./cn_data/dev.tsv', sep='\t')
    # 添加句子长度列
    train_data["sentence_length"] = list(map(lambda x: len(x), train_data["sentence"]))
    dev_data["sentence_length"] = list(map(lambda x: len(x), dev_data["sentence"]))
    # 画散点图
    sns.stripplot(y="sentence_length", x='label', data=train_data, hue='label')
    plt.show()
    sns.stripplot(y="sentence_length", x='label', data=dev_data, hue='label')
    plt.show()
if __name__ == '__main__':
    # dm_content_length()
    dm_stripplot()