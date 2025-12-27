# coding:utf-8
# 导入必备工具包
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def dm_label_sns_countplot():
    # plt.style.use('fivethirtyeight')
    # # 直接获取数据
    # train_data = pd.read_csv('./cn_data/train.tsv', sep='\t')
    # print(f'train_data--》{train_data.head()}')
    # # 统计标签数量分布
    # sns.countplot(x='label', data=train_data)
    # plt.title("train_data")
    # plt.show()
    # 直接获取数据
    dev_data = pd.read_csv('./cn_data/dev.tsv', sep='\t')
    print(f'dev_data--》{dev_data.head()}')
    # 统计标签数量分布
    # sns.countplot(x='label', data=dev_data)
    # x和y只能写一个，要不是x轴显示要不是y轴显示，hue按照哪类分组

    sns.countplot(x='label', data=dev_data, hue='label')
    plt.title("dev_data")
    plt.show()


if __name__ == '__main__':
    dm_label_sns_countplot()
