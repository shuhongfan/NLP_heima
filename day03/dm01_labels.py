# -*- coding:utf-8 -*-
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def dm_label_sns_countplot():

    # 1 设置显示风格plt.style.use('fivethirtyeight')
    plt.style.use('fivethirtyeight')

    # 2 pd.read_csv 读训练集 验证集数据
    train_data = pd.read_csv('./cn_data/train.tsv', sep='\t')
    print(f'train_data--》{train_data}')
    dev_data = pd.read_csv('./cn_data/dev.tsv', sep='\t')

    # 3 sns.countplot() 统计label标签的0、1分组数量
    sns.countplot(y='label', data=train_data, hue='label')

    # 4 画图展示 plt.title() plt.show()
    plt.title('train_label')
    plt.show()

    # 验证集上标签的数量分布
    # 3-2 sns.countplot() 统计label标签的0、1分组数量
    # sns.countplot(x='label', data = dev_data)
    sns.countplot(x=dev_data["label"])

    # 4-2 画图展示 plt.title() plt.show()
    plt.title('dev_label')
    plt.show()




if __name__ == '__main__':
    dm_label_sns_countplot()
